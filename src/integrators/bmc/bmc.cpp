#include <mitsuba/render/scene.h>

MTS_NAMESPACE_BEGIN

class MyIntegrator : public SamplingIntegrator {
public:
    // Initialize the integrator with the specified properties
    MyIntegrator(const Properties &props) : SamplingIntegrator(props) {
        Spectrum defaultColor;
        defaultColor.fromLinearRGB(0.0f, 0.0f, 0.0f);
        m_color = props.getSpectrum("color", defaultColor);
        m_maxDepth = props.getInteger("maxDepth", 1);
    }

    // Unserialize from a binary data stream
    MyIntegrator(Stream *stream, InstanceManager *manager) : SamplingIntegrator(stream, manager) {
        m_color = Spectrum(stream);
        m_maxDist = stream->readFloat();
        m_maxDepth = stream->readInt();
    }

    // Serialize to a binary data stream
    void serialize(Stream *stream, InstanceManager *manager) const {
        SamplingIntegrator::serialize(stream, manager);
        m_color.serialize(stream);
        stream->writeFloat(m_maxDist);
        stream->writeInt(m_maxDepth);
    }

    // Query for an unbiased estimate of the radiance along <tt>r</tt>
    //  Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
    //      const Scene *scene = rRec.scene;
    //      Intersection &its = rRec.its;
    //      RayDifferential ray(r);

    //      Spectrum Li(0.0f);
    //      Spectrum throughput(1.0f);

    //     while (rRec.depth <= m_maxDepth || m_maxDepth < 0) {
    //         // No intersection case, return environment luminaire
    //         if (!scene->rayIntersect(ray, its)) {
    //             Li += throughput * scene->evalEnvironment(ray);
    //             break;
    //         }

    //         // Check if the path length have surpassed the maxDepth
    //         if (rRec.depth >= m_maxDepth && m_maxDepth > 0) {
    //             break;
    //         }

    //         // Direct illumination contribution
    //         // by now let's use the depth integrator
    //         Float distance = its.t;
    //         Li += throughput * Spectrum(1.0f - distance/m_maxDist) * m_color;

    //         // Trace the ray in new direction
    //         const BSDF *bsdf = its.getBSDF(ray);
    //         BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
    //         Spectrum bsdfWeight = bsdf->sample(bRec, rRec.nextSample2D());
    //         if (bsdfWeight.isZero())
    //             break;
                    
    //         // Update beta
    //         throughput *= bsdfWeight;
                
    //         // Set new ray
    //         const Vector wo = its.toWorld(bRec.wo);
    //         ray = RayDifferential(its.p, wo, ray.time);
    //     }

    //     return Li;
    // }
    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
        const Scene *scene = rRec.scene;
        Intersection &its = rRec.its;
        RayDifferential ray(r);
        Spectrum Lout(0.0f);

        if (!scene->rayIntersect(ray, its)) {
            return scene->evalEnvironment(ray);
        }

        // Check if the path length have surpassed the maxDepth
        /* if (rRec.depth >= m_maxDepth && m_maxDepth > 0) {
            return Lout;
        }*/

        const BSDF *bsdf = its.getBSDF(ray);
        
        DirectSamplingRecord dRec(its);
        int nSamples = 500; // number of samples
        Float bsdfPdf;
        for (int i = 0; i < nSamples; i++) {
            // Randomly sample direction leaving surface for random walk (within the hemisphere)
            //BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
            BSDFSamplingRecord bRec(its, its.toLocal(dRec.d), ERadiance);
            // Evaluate BSDF at surface for sampled direction
            Spectrum bsdfValue = bsdf->sample(bRec, bsdfPdf, rRec.nextSample2D());
            //Spectrum bsdfValue = bsdf->eval(bRec);
            const Vector wo = normalize(its.toWorld(bRec.wo));

            // Recursively trace ray to estimate incident radiance at surface
            RayDifferential secondaryRay(its.p, wo, ray.time);
			rRec.depth++;
            //bsdfPdf = 1.0 / (2.0 * 3.14159265358979323846);
            
            Lout += Li(secondaryRay, rRec) * bsdfValue;
        }
        return Lout /= (Float)nSamples;
   }

    // Preprocess function -- called on the initiating machine
    bool preprocess(const Scene *scene, RenderQueue *queue, const RenderJob *job, int sceneResID, int cameraResID, int samplerResID) {
        SamplingIntegrator::preprocess(scene, queue, job, sceneResID, cameraResID, samplerResID);

        const AABB &sceneAABB = scene->getAABB();
        /* Find the camera position at t=0 seconds */
        Point cameraPosition = scene->getSensor()->getWorldTransform()->eval(0).transformAffine(Point(0.0f));
        m_maxDist = - std::numeric_limits<Float>::infinity();

        for (int i=0; i<8; i++)
            m_maxDist = std::max(m_maxDist, (cameraPosition - sceneAABB.getCorner(i)).length());

        return true;
    }

    MTS_DECLARE_CLASS()

private:
    Spectrum m_color;
    Float m_maxDist;
    int m_maxDepth;
};

MTS_IMPLEMENT_CLASS_S(MyIntegrator, false, SamplingIntegrator)
MTS_EXPORT_PLUGIN(MyIntegrator, "A contrived integrator");

MTS_NAMESPACE_END