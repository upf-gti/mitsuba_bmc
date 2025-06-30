#include <mitsuba/render/scene.h>

#include <stdlib.h>

#include "GPRender/src/bmc.h"
#include "GPRender/src/cov_kernels.h"

MTS_NAMESPACE_BEGIN

// Rotate a sample around Y by a random angle
Vector3 rotate_around_y(Vector3 dir, Float alpha) {
    Float sin_alpha = sin(alpha);
    Float cos_alpha = cos(alpha);
    return Vector3(
        dir.x * cos_alpha + dir.z * sin_alpha,
        dir.y,
        -dir.x * sin_alpha + dir.z * cos_alpha
    );
}

// Rotate a sample around Z by a random angle
Vector3 rotate_around_z(Vector3 dir, Float alpha) {
    Float sin_alpha = sin(alpha);
    Float cos_alpha = cos(alpha);
    return Vector3(
        dir.x * cos_alpha - dir.y * sin_alpha,
        dir.x * sin_alpha + dir.y * cos_alpha,
        dir.z
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

// Get random vector on an hemishpere facing the Z axis
Vector3 random_on_hemisphere() {
    Vector3 normal = Vector3(0.0, 0.0, 1.0);
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
        m_maxDepth = props.getInteger("maxDepth", 1);
    }

    // Unserialize from a binary data stream
    BMCIntegrator(Stream *stream, InstanceManager *manager) : SamplingIntegrator(stream, manager) {
        m_maxDepth = stream->readInt();
    }

    // Serialize to a binary data stream
    void serialize(Stream *stream, InstanceManager *manager) const {
        SamplingIntegrator::serialize(stream, manager);
        stream->writeInt(m_maxDepth);
    }

    // Auxiliar radiance computation function only for the indirect component
    Spectrum Li_recursive(const RayDifferential& r, RadianceQueryRecord& rRec) const {
        const Scene* scene = rRec.scene;
        Intersection& its = rRec.its;
        RayDifferential ray(r);
        Spectrum Li(0.0);

        // If there is not intersection, return the environment
        if (!scene->rayIntersect(ray, its)) {
            return scene->evalEnvironment(ray);
        }

        // Check if the path length have surpassed the limit of bounces
        if (rRec.depth > m_maxDepth && m_maxDepth > 0) {
            return Li;
        }

        // Pick a random BMC Gaussian Process
        uint32_t randomGP = rand() % numBMCs;
        BMC<Vector3, Spectrum>* bmc = bmcList[randomGP];

        // Store the radiance of each random direction
        std::vector<Spectrum> radianceSamples;

        // Random angle to rotate the GP directions
        Float alpha = 2.0 * M_PI * rand() / (Float)RAND_MAX;

        // Get surface properties
        const BSDF* bsdf = its.getBSDF(ray);

        // Loop for each random directions computed in the preprocess step
        for (uint32_t sIdx = 0; sIdx < numShadingSamples; sIdx++) {
            Vector3 wo = bmc->get_gaussian_process()->get_observation(sIdx);
            // Rotate to get different directions each sample/pixel/intersection (with same covariance matrix)
            Vector3 woLocal = rotate_around_z(wo, alpha);
            // Rotate to align hemisphere directions to intersection normal
            Vector3 wiLocal = normalize(its.toLocal(-ray.d));

            // Evaluate BSDF at surface for sampled direction
            BSDFSamplingRecord bRec(its, wiLocal, woLocal, ERadiance);
            Spectrum bsdfVal = bsdf->eval(bRec); // reflectance * cosine term

            // Recursively trace ray to estimate incident radiance at surface
            Vector3 woWorld = normalize(its.toWorld(woLocal));
            RayDifferential nextRay(its.p, woWorld, ray.time);
            rRec.depth++;

            // Store each color retrieved from every direction in an array
            radianceSamples.push_back(Li_recursive(nextRay, rRec) * bsdfVal);
        }

        // and use this array to compute the final radiance
        Spectrum Li_ind(0.0);
        bmc->compute_integral(radianceSamples, Li_ind);
        Li += Li_ind;

        return Li;
    }

    // Radiance computation function
    /*
        wi refers to the vector comming from the camera
        wo refers to the reflecting vector
        all vectors should point away from the intersection
    */
    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
        const Scene *scene = rRec.scene;
        Intersection &its = rRec.its;
        RayDifferential ray(r);
        Spectrum Li(0.0);

        // If there is not intersection, return the environment
        if (!scene->rayIntersect(ray, its)) {
            return scene->evalEnvironment(ray);
        }

        // Check if the path length have surpassed the limit of bounces
        if (rRec.depth > m_maxDepth && m_maxDepth > 0) {
            return Li;
        }

        /* ==================================================================== */
        /*                          Emissive radiance                           */
        /* ==================================================================== */

        if (its.isEmitter()) {
            Li += its.Le(-ray.d);
        }

        /* ==================================================================== */
        /*                          Direct radiance                             */
        /* ==================================================================== */

        // TODO
        // ..

        /* ==================================================================== */
        /*                          Indirect radiance                           */
        /* ==================================================================== */

        // Pick a random BMC Gaussian Process
        uint32_t randomGP = rand() % numBMCs;
        BMC<Vector3, Spectrum>* bmc = bmcList[randomGP];

        // Store the radiance of each random direction
        std::vector<Spectrum> radianceSamples;

        // Random angle to rotate the GP directions
        Float alpha = 2.0 * M_PI * rand() / (Float)RAND_MAX;

        // Get surface properties
        const BSDF *bsdf = its.getBSDF(ray);

        // Loop for each random directions computed in the preprocess step
        for (uint32_t sIdx = 0; sIdx < numShadingSamples; sIdx++) {
            Vector3 wo = bmc->get_gaussian_process()->get_observation(sIdx);
            // Rotate to get different directions each sample/pixel/intersection (with same covariance matrix)
            Vector3 woLocal = rotate_around_z(wo, alpha);
            // Rotate to align hemisphere directions to intersection normal
            Vector3 wiLocal = normalize(its.toLocal(-ray.d));

            // Evaluate BSDF at surface for sampled direction
            BSDFSamplingRecord bRec(its, wiLocal, woLocal, ERadiance);
            Spectrum bsdfVal = bsdf->eval(bRec); // reflectance * cosine term

            // Recursively trace ray to estimate incident radiance at surface
            Vector3 woWorld = normalize(its.toWorld(woLocal));
            RayDifferential nextRay(its.p, woWorld, ray.time);
            rRec.depth++;

            // Store each color retrieved from every direction in an array
            radianceSamples.push_back(Li_recursive(nextRay, rRec) * bsdfVal);
        }

        // and use this array to compute the final radiance
        Spectrum Li_ind(0.0);
        bmc->compute_integral(radianceSamples, Li_ind);
        Li += Li_ind;

        return Li;
   }

    // Preprocess function -- called on the initiating machine
    bool preprocess(const Scene *scene, RenderQueue *queue, const RenderJob *job, int32_t sceneResID, int32_t cameraResID, int32_t samplerResID) {
        SamplingIntegrator::preprocess(scene, queue, job, sceneResID, cameraResID, samplerResID);

        // Create multiple Gaussian Process (GP) instances to have way more sampling directions mantaining performance
        bmcList.resize(numBMCs);

        for (uint32_t i = 0; i < numBMCs; ++i) {
            mitsuba_kernel::sSobolevParams sobolevParams;
            sobolevParams.s = 1.5;

            GaussianProcess<Vector3, Spectrum>* gaussianProcess = new GaussianProcess<Vector3, Spectrum>(mitsuba_kernel::sobolev, &sobolevParams, sizeof(mitsuba_kernel::sSobolevParams), 0.01);

            // Set N number of samples (observation/training points), in our case directions
            std::vector<Vector3> sampleDirections;
            sampleDirections.reserve(numShadingSamples);

            // Victor's birth year plus offset :)
            srand(1998 + i);

            // Generate and store N random directions in the hemisphere
            for (uint32_t i = 0; i < numShadingSamples; i++) {
                sampleDirections.push_back(random_on_hemisphere());
            }

            // Fill the GP instance with the array of directions (observation points)
            gaussianProcess->set_observations(sampleDirections, {});

            bmcList[i] = new BMC<Vector3, Spectrum>(random_on_hemisphere, gaussianProcess);
        }

        return true;
    }

    MTS_DECLARE_CLASS()

private:
    uint32_t m_maxDepth;

    std::vector<BMC<Vector, Spectrum>*> bmcList;
    uint32_t numBMCs = 1;
    // Number of cached sample directions in the hemisphere
    uint32_t numShadingSamples = 200;
};

MTS_IMPLEMENT_CLASS_S(BMCIntegrator, false, SamplingIntegrator)
MTS_EXPORT_PLUGIN(BMCIntegrator, "Bayesian Monte Carlo integrator");

MTS_NAMESPACE_END