// SPDX-License-Identifier: MIT
// USD mesh shape plugin for Mitsuba 3
//
// Loads a mesh from a USD (.usd/.usda/.usdc/.usdz) file using tinyusdz.
// The plugin reads a specific mesh prim identified by `prim_path`.
// If `prim_path` is not specified, it loads the first mesh found.
//
// Scene-level orchestration (materials, lights, cameras, hierarchy
// transforms) is handled by the Python-side converter (render_usd.py),
// which passes `to_world` for each mesh.

#include <mitsuba/render/mesh.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/timer.h>
#include <mitsuba/core/profiler.h>

// tinyusdz
#include "tinyusdz.hh"
#include "usdGeom.hh"
#include "tydra/scene-access.hh"
#include "usd_compose.h"

#include <unordered_map>
#include <memory>
#include <mutex>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class USDMesh final : public Mesh<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Mesh, m_name, m_bbox, m_to_world, m_vertex_count,
                   m_face_count, m_vertex_positions, m_vertex_normals,
                   m_vertex_texcoords, m_faces, m_face_normals,
                   recompute_vertex_normals, has_vertex_normals, initialize)
    MI_IMPORT_TYPES()

    using typename Base::ScalarSize;
    using typename Base::ScalarIndex;
    using typename Base::InputFloat;
    using typename Base::InputPoint3f;
    using typename Base::InputVector2f;
    using typename Base::InputVector3f;
    using typename Base::InputNormal3f;
    using typename Base::FloatStorage;

    USDMesh(const Properties &props) : Base(props) {
        auto fr = file_resolver();
        fs::path file_path = fr->resolve(props.get<std::string_view>("filename"));
        std::string prim_path = props.get<std::string>("prim_path", "");

        m_name = file_path.filename().string();
        if (!prim_path.empty())
            m_name += ":" + prim_path;

        auto fail = [&](const char *descr, auto... args) {
            Throw(("Error loading USD mesh \"%s\": " + std::string(descr))
                      .c_str(), m_name, args...);
        };

        Log(Debug, "Loading USD mesh from \"%s\" (prim: %s) ..",
            file_path.string(), prim_path.empty() ? "<first>" : prim_path);

        if (!fs::exists(file_path))
            fail("file not found");

        ScopedPhase phase(ProfilerPhase::LoadGeometry);
        Timer timer;

        // --- Load composed USD Stage (cached across instances) ---
        std::string abs_path = fs::absolute(file_path).string();
        std::string base_dir = fs::absolute(file_path).parent_path().string();
        std::string load_err;
        auto stage_ptr = usd_compose::load_composed_stage(abs_path, base_dir, &load_err);
        if (!stage_ptr)
            fail("%s", load_err.c_str());
        const tinyusdz::Stage &stage = *stage_ptr;

        // --- Find mesh prim ---
        const tinyusdz::GeomMesh *mesh = nullptr;

        if (!prim_path.empty()) {
            auto result = stage.GetPrimAtPath(tinyusdz::Path(prim_path, ""));
            if (!result)
                fail("prim not found: %s", prim_path.c_str());
            mesh = result.value()->as<tinyusdz::GeomMesh>();
            if (!mesh)
                fail("prim is not a Mesh: %s", prim_path.c_str());
        } else {
            // Find first mesh via Tydra
            std::map<std::string, const tinyusdz::GeomMesh *> meshes;
            tinyusdz::tydra::ListPrims(stage, meshes);
            if (meshes.empty())
                fail("no mesh found in file");
            mesh = meshes.begin()->second;
        }

        // --- Extract raw geometry ---
        auto points = mesh->get_points();
        auto face_indices = mesh->get_faceVertexIndices();
        auto face_counts = mesh->get_faceVertexCounts();

        if (points.empty())   fail("mesh has no vertices");
        if (face_indices.empty()) fail("mesh has no face indices");
        if (face_counts.empty())  fail("mesh has no face counts");

        // Normals
        auto normals = mesh->get_normals();
        auto norm_interp = mesh->get_normalsInterpolation();
        bool has_normals = !normals.empty() && !m_face_normals;

        // UVs - try common primvar names
        std::vector<tinyusdz::value::texcoord2f> uvs;
        tinyusdz::Interpolation uv_interp = tinyusdz::Interpolation::Vertex;
        {
            tinyusdz::GeomPrimvar pv;
            std::string pv_err;
            for (const char *name : {"st", "UVMap", "uv"}) {
                if (mesh->get_primvar(name, &pv, &pv_err)) {
                    pv.flatten_with_indices(&uvs);
                    uv_interp = pv.get_interpolation();
                    break;
                }
            }
        }
        bool has_uvs = !uvs.empty();

        // Determine if we need to "unshare" vertices
        // (needed when any attribute uses faceVarying interpolation)
        bool fv_normals = has_normals &&
                          norm_interp == tinyusdz::Interpolation::FaceVarying;
        bool fv_uvs = has_uvs &&
                      uv_interp == tinyusdz::Interpolation::FaceVarying;
        bool need_unshare = fv_normals || fv_uvs;

        // --- Triangulate ---
        // Count triangles from face vertex counts
        size_t total_tris = 0;
        for (int32_t fc : face_counts) {
            if (fc < 3) fail("degenerate face with %d vertices", fc);
            total_tris += (size_t)(fc - 2);
        }

        if (need_unshare) {
            // Unshared: one vertex per triangle-corner
            build_unshared(points, face_indices, face_counts,
                           has_normals ? &normals : nullptr, norm_interp,
                           has_uvs ? &uvs : nullptr, uv_interp,
                           total_tris, fail);
        } else {
            // Shared: vertices indexed by position index
            build_shared(points, face_indices, face_counts,
                         has_normals ? &normals : nullptr,
                         has_uvs ? &uvs : nullptr,
                         total_tris, fail);
        }

        if (!has_normals && !m_face_normals) {
            Timer timer2;
            recompute_vertex_normals();
            Log(Debug, "\"%s\": computed vertex normals (took %s)", m_name,
                util::time_string((float) timer2.value()));
        }

        size_t vb = 3 * sizeof(InputFloat);
        if (has_normals && !m_face_normals) vb += 3 * sizeof(InputFloat);
        if (has_uvs) vb += 2 * sizeof(InputFloat);
        Log(Debug, "\"%s\": read %i faces, %i vertices (%s in %s)",
            m_name, m_face_count, m_vertex_count,
            util::mem_string(m_face_count * 3 * sizeof(ScalarIndex) +
                             m_vertex_count * vb),
            util::time_string((float) timer.value()));

        initialize();
    }

private:
    template <typename FailFn>
    void build_shared(
        const std::vector<tinyusdz::value::point3f> &points,
        const std::vector<int32_t> &face_indices,
        const std::vector<int32_t> &face_counts,
        const std::vector<tinyusdz::value::normal3f> *normals,
        const std::vector<tinyusdz::value::texcoord2f> *uvs,
        size_t total_tris,
        FailFn &fail)
    {
        m_vertex_count = (ScalarSize) points.size();
        m_face_count = (ScalarSize) total_tris;

        std::unique_ptr<float[]> vpos(new float[m_vertex_count * 3]);
        std::unique_ptr<float[]> vnorm(new float[m_vertex_count * 3]());
        std::unique_ptr<float[]> vtex;
        if (uvs) vtex.reset(new float[m_vertex_count * 2]);

        // Copy vertices (apply to_world)
        for (size_t i = 0; i < m_vertex_count; ++i) {
            InputPoint3f p(points[i].x, points[i].y, points[i].z);
            p = m_to_world.scalar() * p;
            if (unlikely(!all(dr::isfinite(p))))
                fail("invalid vertex position at index %zu", i);
            m_bbox.expand(p);
            dr::store(vpos.get() + i * 3, p);
        }

        // Normals (vertex interpolation: indexed by vertex index)
        if (normals) {
            for (size_t i = 0; i < m_vertex_count && i < normals->size(); ++i) {
                InputNormal3f n((*normals)[i].x, (*normals)[i].y,
                                (*normals)[i].z);
                n = dr::normalize(m_to_world.scalar() * n);
                dr::store(vnorm.get() + i * 3, n);
            }
        }

        // UVs (vertex interpolation: indexed by vertex index)
        if (uvs) {
            for (size_t i = 0; i < m_vertex_count && i < uvs->size(); ++i) {
                dr::store(vtex.get() + i * 2,
                          InputVector2f((*uvs)[i].s, (*uvs)[i].t));
            }
        }

        // Triangulate faces (fan triangulation)
        std::vector<ScalarIndex> tris(total_tris * 3);
        size_t tri_idx = 0, fv = 0;
        for (int32_t fc : face_counts) {
            for (int32_t k = 1; k < fc - 1; ++k) {
                tris[tri_idx++] = (ScalarIndex) face_indices[fv];
                tris[tri_idx++] = (ScalarIndex) face_indices[fv + k];
                tris[tri_idx++] = (ScalarIndex) face_indices[fv + k + 1];
            }
            fv += (size_t) fc;
        }

        m_faces = dr::load<DynamicBuffer<UInt32>>(tris.data(), m_face_count * 3);
        m_vertex_positions = dr::load<FloatStorage>(vpos.get(),
                                                    m_vertex_count * 3);
        m_vertex_normals = dr::load<FloatStorage>(vnorm.get(),
                                                  m_vertex_count * 3);
        if (uvs)
            m_vertex_texcoords = dr::load<FloatStorage>(vtex.get(),
                                                        m_vertex_count * 2);
    }

    template <typename FailFn>
    void build_unshared(
        const std::vector<tinyusdz::value::point3f> &points,
        const std::vector<int32_t> &face_indices,
        const std::vector<int32_t> &face_counts,
        const std::vector<tinyusdz::value::normal3f> *normals,
        tinyusdz::Interpolation norm_interp,
        const std::vector<tinyusdz::value::texcoord2f> *uvs,
        tinyusdz::Interpolation uv_interp,
        size_t total_tris,
        FailFn &fail)
    {
        // Each triangle corner gets its own vertex
        m_vertex_count = (ScalarSize)(total_tris * 3);
        m_face_count = (ScalarSize) total_tris;

        std::unique_ptr<float[]> vpos(new float[m_vertex_count * 3]);
        std::unique_ptr<float[]> vnorm(new float[m_vertex_count * 3]());
        std::unique_ptr<float[]> vtex;
        if (uvs) vtex.reset(new float[m_vertex_count * 2]);
        std::vector<ScalarIndex> tris(m_face_count * 3);

        ScalarIndex vert_idx = 0;
        size_t fv = 0;  // face-vertex offset into face_indices

        for (int32_t fc : face_counts) {
            for (int32_t k = 1; k < fc - 1; ++k) {
                // Three corner face-vertex offsets for this triangle
                size_t corners[3] = { fv, fv + (size_t)k, fv + (size_t)k + 1 };

                for (int c = 0; c < 3; ++c) {
                    size_t fv_off = corners[c];
                    int32_t vi = face_indices[fv_off];

                    // Position
                    InputPoint3f p(points[vi].x, points[vi].y, points[vi].z);
                    p = m_to_world.scalar() * p;
                    if (unlikely(!all(dr::isfinite(p))))
                        fail("invalid vertex position");
                    m_bbox.expand(p);
                    dr::store(vpos.get() + vert_idx * 3, p);

                    // Normal
                    if (normals) {
                        size_t ni = (norm_interp ==
                                     tinyusdz::Interpolation::FaceVarying)
                                        ? fv_off : (size_t) vi;
                        if (ni < normals->size()) {
                            InputNormal3f n((*normals)[ni].x,
                                            (*normals)[ni].y,
                                            (*normals)[ni].z);
                            n = dr::normalize(m_to_world.scalar() * n);
                            dr::store(vnorm.get() + vert_idx * 3, n);
                        }
                    }

                    // UV
                    if (uvs) {
                        size_t ui = (uv_interp ==
                                     tinyusdz::Interpolation::FaceVarying)
                                        ? fv_off : (size_t) vi;
                        if (ui < uvs->size()) {
                            dr::store(vtex.get() + vert_idx * 2,
                                      InputVector2f((*uvs)[ui].s,
                                                    (*uvs)[ui].t));
                        }
                    }

                    tris[vert_idx] = vert_idx;
                    vert_idx++;
                }
            }
            fv += (size_t) fc;
        }

        m_faces = dr::load<DynamicBuffer<UInt32>>(tris.data(), m_face_count * 3);
        m_vertex_positions = dr::load<FloatStorage>(vpos.get(),
                                                    m_vertex_count * 3);
        m_vertex_normals = dr::load<FloatStorage>(vnorm.get(),
                                                  m_vertex_count * 3);
        if (uvs)
            m_vertex_texcoords = dr::load<FloatStorage>(vtex.get(),
                                                        m_vertex_count * 2);
    }

public:
    MI_DECLARE_CLASS(USDMesh)
    MI_TRAVERSE_CB(Base)
};

MI_EXPORT_PLUGIN(USDMesh)
NAMESPACE_END(mitsuba)
