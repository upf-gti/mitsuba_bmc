#include <mitsuba/render/scene.h>

#include <stdlib.h>

#include "GPRender/src/bmc.h"
#include "GPRender/src/cov_kernels.h"

MTS_NAMESPACE_BEGIN

// Rotate a sample arround Y by a random angle
Vector3 rotate_around_y(Vector3 dir, Float alpha) {
    Float sin_alpha = sin(alpha);
    Float cos_alpha = cos(alpha);
    return Vector3(
        dir.x * cos_alpha + dir.z * sin_alpha,
        dir.y,
        -dir.x * sin_alpha + dir.z * cos_alpha
    );
}

// Get a random Vector3 between a specified scalar range
Vector3 random(Float min, Float max) {
    Float x = min + (max - min) * (rand() / (RAND_MAX + 1.0));
    Float y = min + (max - min) * (rand() / (RAND_MAX + 1.0));
    Float z = min + (max - min) * (rand() / (RAND_MAX + 1.0));
    return Vector3(x, y, z);
};

// Get a random unit vector
Vector3 random_unit_vector() {
    while (true) {
        Vector3 p = random(-1.0, 1.0);
        Float len_sq = p.lengthSquared();
        if (Epsilon < len_sq && len_sq <= 1.0) {
            return p / sqrt(len_sq);
        }
    }
}

// Get random vector on an hemishpere facing the Y axis
Vector3 random_on_hemisphere() {
    Vector3 normal = Vector3(0.0, 1.0, 0.0);
    Vector3 on_unit_sphere = random_unit_vector();
    if (dot(on_unit_sphere, normal) > 0.0) { // In the same hemisphere as the normal
        return on_unit_sphere;
    }
    else {
        return -on_unit_sphere;
    }
}

class BMCIntegrator : public SamplingIntegrator {
public:
    // Initialize the integrator with the specified properties
    BMCIntegrator(const Properties &props) : SamplingIntegrator(props) {
        Spectrum defaultColor;
        defaultColor.fromLinearRGB(0.0, 0.0, 0.0);
        m_color = props.getSpectrum("color", defaultColor);
        m_maxDepth = props.getInteger("maxDepth", 1);
    }

    // Unserialize from a binary data stream
    BMCIntegrator(Stream *stream, InstanceManager *manager) : SamplingIntegrator(stream, manager) {
        m_color = Spectrum(stream);
        m_maxDepth = stream->readInt();
    }

    // Serialize to a binary data stream
    void serialize(Stream *stream, InstanceManager *manager) const {
        SamplingIntegrator::serialize(stream, manager);
        m_color.serialize(stream);
        stream->writeInt(m_maxDepth);
    }

    Spectrum Li_recursive(const RayDifferential& r, RadianceQueryRecord& rRec) const {
        const Scene* scene = rRec.scene;
        Intersection& its = rRec.its;
        RayDifferential ray(r);
        Spectrum L_out(0.0);

        if (!scene->rayIntersect(ray, its)) {
            return scene->evalEnvironment(ray);
        }

        return L_out;
    }

    // Radiance computation function
    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
        const Scene *scene = rRec.scene;
        Intersection &its = rRec.its;
        RayDifferential ray(r);
        Spectrum L_out(0.0);

        if (!scene->rayIntersect(ray, its)) {
            return scene->evalEnvironment(ray);
        }

        // Check if the path length have surpassed the maxDepth
        /* if (rRec.depth >= m_maxDepth && m_maxDepth > 0) {
            return L_out;
        }*/

        // Pick a random BMC Gaussian Process
        uint32_t random_gp = rand() % num_bmcs;
        BMC<Vector3, Spectrum>* bmc = bmc_list[random_gp];

        Float alpha = 2.0 * M_PI * rand() / (Float)RAND_MAX;

        const BSDF *bsdf = its.getBSDF(ray);
        DirectSamplingRecord dRec(its);
        std::vector<Spectrum> radiance_samples;

        for (uint32_t s_idx = 0; s_idx < num_shading_samples; s_idx++) {
            // In computeColor use the previously stored directions in GP to define the reflected ray directions
            Vector3 wo = bmc->get_gaussian_process()->get_observation(s_idx);
            // Rotate to get different directions each sample (with same covariance matrix)
            Vector3 wo2 = rotate_around_y(wo, alpha);
            // Rotate to align hemisphere directions to intersection normal
            Vector3 wo3 = normalize(its.toWorld(wo2));

            // Evaluate BSDF at surface for sampled direction
            //Vector3 fcos = its_ctx.shape->material->evaluateBSDF(its_ctx.normal, wo, wi) * abs(dot(its_ctx.normal, wi));
            BSDFSamplingRecord bRec(its, -ray.d, wo3, ERadiance);
            // Evaluate BSDF at surface for sampled direction
            Spectrum bsdf_value = bsdf->sample(bRec, rRec.nextSample2D());
            bRec.wo = wo3;

            // Recursively trace ray to estimate incident radiance at surface
            RayDifferential secondaryRay(its.p, its.geoFrame.n, ray.time);
            rRec.depth++;

            // In this basic example, with only 1 depth (light from environment), store each color retrieved from every direction in an array
            radiance_samples.push_back(Li_recursive(secondaryRay, rRec));// *bsdf_value);
        }

        // and use this array to compute the final radiance
        L_out = bmc->compute_integral(radiance_samples);

        return radiance_samples[0];
   }

    // Preprocess function -- called on the initiating machine
    bool preprocess(const Scene *scene, RenderQueue *queue, const RenderJob *job, int32_t sceneResID, int32_t cameraResID, int32_t samplerResID) {
        SamplingIntegrator::preprocess(scene, queue, job, sceneResID, cameraResID, samplerResID);

        // Create multiple Gaussian Process (GP) instances to have way more sampling directions mantaining performance
        bmc_list.resize(num_bmcs);

        for (uint32_t i = 0; i < num_bmcs; ++i) {
            mitsuba_kernel::sSobolevParams sobolev_params;
            sobolev_params.s = 1.5;

            GaussianProcess<Vector3, Spectrum>* gaussian_process = new GaussianProcess<Vector3, Spectrum>(mitsuba_kernel::sobolev, &sobolev_params, sizeof(mitsuba_kernel::sSobolevParams), 0.01);

            // Set x number of samples (observation/training points), in our case directions
            std::vector<Vector3> sample_directions;
            sample_directions.reserve(num_shading_samples);

            // Victor's birth year plus offset :)
            srand(1998 + i);

            // Generate x random directions in sphere and store in array
            for (uint32_t s_idx = 0; s_idx < num_shading_samples; s_idx++) {
                sample_directions.push_back(random_on_hemisphere());
            }

            // Fill the GP instance with the array of directions (observation points)
            gaussian_process->set_observations(sample_directions, {});

            bmc_list[i] = new BMC<Vector3, Spectrum>(random_on_hemisphere, gaussian_process);
        }

        return true;
    }

    MTS_DECLARE_CLASS()

private:
    Spectrum m_color;
    uint32_t m_maxDepth;

    std::vector<BMC<Vector, Spectrum>*> bmc_list;
    uint32_t num_bmcs = 1;
    // Number of cached sample directions in the hemisphere
    uint32_t num_shading_samples = 10;
};

MTS_IMPLEMENT_CLASS_S(BMCIntegrator, false, SamplingIntegrator)
MTS_EXPORT_PLUGIN(BMCIntegrator, "A contrived integrator");

MTS_NAMESPACE_END