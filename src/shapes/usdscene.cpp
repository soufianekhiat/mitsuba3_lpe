// SPDX-License-Identifier: MIT
// USD scene shape plugin for Mitsuba 3
//
// Loads an entire USD file and expands into one Mesh shape per GeomMesh prim,
// each with its own OpenPBR BSDF (from MaterialX open_pbr_surface, material
// binding, displayColor, or default grey).
// World transforms are computed by traversing the prim hierarchy.
// Supports MaterialX texture-driven inputs and USD lights.

#include <mitsuba/render/mesh.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/timer.h>
#include <mitsuba/core/profiler.h>

// tinyusdz
#include "tinyusdz.hh"
#include "usdGeom.hh"
#include "usdShade.hh"
#include "usdLux.hh"
#include "xform.hh"
#include "usd_compose.h"

#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <unordered_map>

NAMESPACE_BEGIN(mitsuba)

// ---------------------------------------------------------------------------
// MaterialX open_pbr_surface parser — constants + textures
// ---------------------------------------------------------------------------

/// Parsed OpenPBR parameters from a .mtlx file.
/// For each texturable parameter, stores either a constant value or a
/// resolved texture file path (absolute).  Texture takes precedence.
struct MtlxOpenPBRParams {
    bool valid = false;

    // base
    float base_weight = 1.0f;
    std::array<float,3> base_color = {0.8f, 0.8f, 0.8f};
    bool has_base_color = false;
    std::string base_color_tex;
    bool base_color_srgb = true;

    float base_metalness = 0.0f;
    bool has_metalness = false;
    std::string base_metalness_tex;

    // specular
    float specular_weight = 1.0f;
    bool has_specular_weight = false;
    std::array<float,3> specular_color = {1.0f, 1.0f, 1.0f};
    bool has_specular_color = false;
    std::string specular_color_tex;
    float specular_roughness = 0.3f;
    bool has_specular_roughness = false;
    std::string specular_roughness_tex;
    float specular_ior = 1.5f;
    bool has_specular_ior = false;

    // transmission
    float transmission_weight = 0.0f;
    bool has_transmission = false;
    std::string transmission_weight_tex;
    std::array<float,3> transmission_color = {1.0f, 1.0f, 1.0f};
    bool has_transmission_color = false;
    std::string transmission_color_tex;

    // geometry
    bool thin_walled = false;
    bool has_thin_walled = false;

    // coat
    float coat_weight = 0.0f;
    bool has_coat = false;
    std::string coat_weight_tex;
    std::array<float,3> coat_color = {1.0f, 1.0f, 1.0f};
    bool has_coat_color = false;
    std::string coat_color_tex;
    float coat_roughness = 0.0f;
    bool has_coat_roughness = false;
    std::string coat_roughness_tex;
    float coat_ior = 1.5f;
    bool has_coat_ior = false;
};

/// Info about a MaterialX <image> node
struct MtlxImageNode {
    std::string name;       // node name (e.g. "BaseColor")
    std::string file;       // texture file path (relative, may contain <UDIM>)
    std::string type;       // "color3", "float", "vector3"
    std::string colorspace; // "srgb_tx", "Raw", etc.
};

// --- XML helpers ---

static std::string read_file_to_string(const std::string &path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) return {};
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

static std::string trim(const std::string &s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    if (a == std::string::npos) return {};
    return s.substr(a, b - a + 1);
}

static std::string xml_attr(const std::string &elem, const std::string &attr) {
    std::string needle = attr + "=\"";
    auto pos = elem.find(needle);
    if (pos == std::string::npos) return {};
    pos += needle.size();
    auto end = elem.find('"', pos);
    if (end == std::string::npos) return {};
    return elem.substr(pos, end - pos);
}

static float parse_float(const std::string &s, float fallback = 0.0f) {
    try { return std::stof(s); }
    catch (...) { return fallback; }
}

static std::array<float,3> parse_color3(const std::string &s) {
    std::array<float,3> out = {0.5f, 0.5f, 0.5f};
    std::string clean = s;
    for (char &c : clean) if (c == ',') c = ' ';
    std::istringstream iss(clean);
    iss >> out[0] >> out[1] >> out[2];
    return out;
}

/// Prefer .png over .tif/.tiff since Mitsuba doesn't support TIFF.
/// If a .png version exists alongside the .tif, always use it.
static std::string try_alt_extension(const std::string &path) {
    auto dot = path.rfind('.');
    if (dot != std::string::npos) {
        std::string ext = path.substr(dot);
        if (ext == ".tif" || ext == ".tiff" || ext == ".TIF") {
            std::string alt = path.substr(0, dot) + ".png";
            if (fs::exists(alt))
                return alt;
        }
    }
    return path;
}

/// Resolve <UDIM> in a texture path to the first existing tile.
static std::string resolve_udim(const std::string &path, const std::string &scene_dir) {
    auto udim_pos = path.find("<UDIM>");
    if (udim_pos == std::string::npos) {
        std::string full = scene_dir + "/" + path;
        return try_alt_extension(full);
    }

    // Try common UDIM tiles
    for (int tile = 1001; tile <= 1010; ++tile) {
        std::string resolved = path;
        resolved.replace(udim_pos, 6, std::to_string(tile));
        std::string full = scene_dir + "/" + resolved;
        std::string alt = try_alt_extension(full);
        if (fs::exists(alt))
            return alt;
    }
    // Fallback: use 1001 even if not found
    std::string resolved = path;
    resolved.replace(udim_pos, 6, "1001");
    return try_alt_extension(scene_dir + "/" + resolved);
}

/// Parse all <image> nodes from a MaterialX XML string.
/// Returns a map: node_name -> MtlxImageNode
static std::unordered_map<std::string, MtlxImageNode>
parse_mtlx_image_nodes(const std::string &xml) {
    std::unordered_map<std::string, MtlxImageNode> images;
    size_t pos = 0;
    while (true) {
        pos = xml.find("<image ", pos);
        if (pos == std::string::npos) break;

        // Find the end of this element (could be multi-line)
        // Look for </image> or self-closing />
        auto close_tag = xml.find("</image>", pos);
        auto self_close = xml.find("/>", pos);
        size_t end_pos;
        if (close_tag != std::string::npos &&
            (self_close == std::string::npos || close_tag < self_close))
            end_pos = close_tag + 8;
        else if (self_close != std::string::npos)
            end_pos = self_close + 2;
        else
            break;

        std::string block = xml.substr(pos, end_pos - pos);

        // Get image node attributes from the opening tag
        auto tag_end = block.find('>');
        std::string tag = block.substr(0, tag_end + 1);
        std::string name = xml_attr(tag, "name");
        std::string type = xml_attr(tag, "type");

        if (!name.empty()) {
            MtlxImageNode img;
            img.name = name;
            img.type = type;

            // Find <input name="file" ... value="..." colorspace="..." />
            size_t inp_pos = 0;
            while (true) {
                inp_pos = block.find("<input", inp_pos);
                if (inp_pos == std::string::npos) break;
                auto inp_end = block.find(">", inp_pos);
                if (inp_end == std::string::npos) break;
                std::string inp = block.substr(inp_pos, inp_end - inp_pos + 1);
                if (xml_attr(inp, "name") == "file") {
                    img.file = xml_attr(inp, "value");
                    img.colorspace = xml_attr(inp, "colorspace");
                }
                inp_pos = inp_end + 1;
            }
            if (!img.file.empty())
                images[name] = img;
        }
        pos = end_pos;
    }
    return images;
}

/// Parse a .mtlx file and extract open_pbr_surface inputs.
/// Resolves texture references to absolute file paths.
static MtlxOpenPBRParams parse_mtlx_openpbr(const std::string &mtlx_path,
                                              const std::string &scene_dir) {
    MtlxOpenPBRParams params;
    std::string xml = read_file_to_string(mtlx_path);
    if (xml.empty()) return params;

    auto openpbr_pos = xml.find("<open_pbr_surface");
    if (openpbr_pos == std::string::npos) return params;

    // Parse image nodes
    auto images = parse_mtlx_image_nodes(xml);

    // Find the open_pbr_surface block
    auto close_pos = xml.find("</open_pbr_surface>", openpbr_pos);
    if (close_pos == std::string::npos) {
        close_pos = xml.find("/>", openpbr_pos);
        if (close_pos == std::string::npos) return params;
    }

    params.valid = true;
    std::string block = xml.substr(openpbr_pos, close_pos - openpbr_pos + 20);

    // Extract all <input> elements
    size_t search_pos = 0;
    while (true) {
        auto inp_pos = block.find("<input", search_pos);
        if (inp_pos == std::string::npos) break;

        auto inp_end = block.find(">", inp_pos);
        if (inp_end == std::string::npos) break;
        std::string inp = block.substr(inp_pos, inp_end - inp_pos + 1);

        std::string name = xml_attr(inp, "name");
        std::string type = xml_attr(inp, "type");
        std::string value = xml_attr(inp, "value");
        std::string nodename = xml_attr(inp, "nodename");

        if (name.empty()) { search_pos = inp_end + 1; continue; }

        // Try to resolve texture reference
        std::string tex_path;
        bool is_srgb = false;
        if (!nodename.empty()) {
            auto img_it = images.find(nodename);
            if (img_it != images.end()) {
                tex_path = resolve_udim(img_it->second.file, scene_dir);
                std::string cs = img_it->second.colorspace;
                // srgb_tx or srgb -> sRGB color data
                is_srgb = (cs.find("srgb") != std::string::npos ||
                           cs.find("sRGB") != std::string::npos);
            }
        }

        // Map to params
        if (name == "base_weight") {
            if (!value.empty()) params.base_weight = parse_float(value, 1.0f);
        } else if (name == "base_color") {
            if (!tex_path.empty()) {
                params.base_color_tex = tex_path;
                params.base_color_srgb = is_srgb;
                params.has_base_color = true;
            } else if (!value.empty() && type == "color3") {
                params.base_color = parse_color3(value);
                params.has_base_color = true;
            }
        } else if (name == "base_metalness") {
            if (!tex_path.empty()) {
                params.base_metalness_tex = tex_path;
                params.has_metalness = true;
            } else if (!value.empty()) {
                params.base_metalness = parse_float(value, 0.0f);
                params.has_metalness = true;
            }
        } else if (name == "specular_weight") {
            if (!value.empty()) {
                params.specular_weight = parse_float(value, 1.0f);
                params.has_specular_weight = true;
            }
        } else if (name == "specular_color") {
            if (!tex_path.empty()) {
                params.specular_color_tex = tex_path;
                params.has_specular_color = true;
            } else if (!value.empty() && type == "color3") {
                params.specular_color = parse_color3(value);
                params.has_specular_color = true;
            }
        } else if (name == "specular_roughness") {
            if (!tex_path.empty()) {
                params.specular_roughness_tex = tex_path;
                params.has_specular_roughness = true;
            } else if (!value.empty()) {
                params.specular_roughness = parse_float(value, 0.3f);
                params.has_specular_roughness = true;
            }
        } else if (name == "specular_ior") {
            if (!value.empty()) {
                params.specular_ior = parse_float(value, 1.5f);
                params.has_specular_ior = true;
            }
        } else if (name == "transmission_weight") {
            if (!tex_path.empty()) {
                params.transmission_weight_tex = tex_path;
                params.has_transmission = true;
            } else if (!value.empty()) {
                params.transmission_weight = parse_float(value, 0.0f);
                params.has_transmission = true;
            }
        } else if (name == "transmission_color") {
            if (!tex_path.empty()) {
                params.transmission_color_tex = tex_path;
                params.has_transmission_color = true;
            } else if (!value.empty() && type == "color3") {
                params.transmission_color = parse_color3(value);
                params.has_transmission_color = true;
            }
        } else if (name == "geometry_thin_walled") {
            if (!value.empty()) {
                params.thin_walled = (value == "true" || value == "1");
                params.has_thin_walled = true;
            }
        } else if (name == "coat_weight") {
            if (!tex_path.empty()) {
                params.coat_weight_tex = tex_path;
                params.has_coat = true;
            } else if (!value.empty()) {
                params.coat_weight = parse_float(value, 0.0f);
                params.has_coat = true;
            }
        } else if (name == "coat_color") {
            if (!tex_path.empty()) {
                params.coat_color_tex = tex_path;
                params.has_coat_color = true;
            } else if (!value.empty() && type == "color3") {
                params.coat_color = parse_color3(value);
                params.has_coat_color = true;
            }
        } else if (name == "coat_roughness") {
            if (!tex_path.empty()) {
                params.coat_roughness_tex = tex_path;
                params.has_coat_roughness = true;
            } else if (!value.empty()) {
                params.coat_roughness = parse_float(value, 0.0f);
                params.has_coat_roughness = true;
            }
        } else if (name == "coat_ior") {
            if (!value.empty()) {
                params.coat_ior = parse_float(value, 1.5f);
                params.has_coat_ior = true;
            }
        }

        search_pos = inp_end + 1;
    }

    return params;
}

// ---------------------------------------------------------------------------

template <typename Float, typename Spectrum>
class USDScene final : public Shape<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Shape)
    MI_IMPORT_TYPES(BSDF, Texture, Emitter)

    USDScene(const Properties &props) {
        auto fr = file_resolver();
        fs::path file_path = fr->resolve(props.get<std::string_view>("filename"));

        if (!fs::exists(file_path))
            Throw("USDScene: file not found: %s", file_path.string());

        ScopedPhase phase(ProfilerPhase::LoadGeometry);
        Timer timer;

        Log(Info, "Loading USD scene from \"%s\" ..", file_path.string());

        std::string abs_filename = fs::absolute(file_path).string();
        std::string base_dir = fs::absolute(file_path).parent_path().string();
        m_base_dir = base_dir;

        // --- Load composed stage via shared helper ---
        std::string load_err;
        auto stage_ptr = usd_compose::load_composed_stage(
            abs_filename, base_dir, &load_err,
            [](const std::string &msg) {
                // Can't use Log() here (outside Mitsuba class context).
            });
        if (!stage_ptr)
            Throw("USDScene: %s", load_err.c_str());
        const tinyusdz::Stage &stage = *stage_ptr;

        // --- Phase 1: Traverse prim tree, collect mesh + light info ---
        std::vector<MeshInfo> mesh_infos;
        std::vector<LightInfo> light_infos;
        {
            auto identity = tinyusdz::value::matrix4d::identity();
            for (const auto &root_prim : stage.root_prims())
                collect_prims(root_prim, identity, stage,
                              mesh_infos, light_infos);
        }

        Log(Info, "USDScene: found %zu meshes, %zu lights",
            mesh_infos.size(), light_infos.size());

        // --- Create light emitters ---
        for (auto &li : light_infos) {
            try {
                auto obj = create_emitter(li);
                if (obj) {
                    auto expanded = obj->expand();
                    if (!expanded.empty())
                        obj = expanded[0];
                    // RectLights are shapes with attached emitters
                    if (obj->type() == ObjectType::Shape)
                        m_shapes.push_back(obj);
                    else
                        m_emitters.push_back(obj);
                }
            } catch (const std::exception &e) {
                Log(Warn, "USDScene: failed to create %s light \"%s\": %s",
                    li.type.c_str(), li.name.c_str(), e.what());
            }
        }

        // Release stage cache before creating usdmesh shapes
        // (usdmesh will re-load into its own DLL-local cache).
        usd_compose::clear_cache();

        // --- Phase 2: Create shapes from collected mesh info ---
        size_t mesh_count = 0;
        for (auto &mi : mesh_infos) {
            try {
                const MtlxOpenPBRParams *mtlx = lookup_mtlx(mi.mat_name);
                auto bsdf = create_openpbr_bsdf(mtlx, mi.color);

                Properties mesh_props;
                mesh_props.set_plugin_name("usdmesh");
                mesh_props.set("filename", abs_filename);
                mesh_props.set("prim_path", mi.prim_path);
                mesh_props.set("to_world", to_mi_transform(mi.world_matrix));
                mesh_props.set("bsdf", bsdf);

                auto shape = PluginManager::instance()->create_object(
                    mesh_props, Shape::Variant, ObjectType::Shape);
                m_shapes.push_back(shape);
                mesh_count++;
            } catch (const std::exception &e) {
                Log(Warn, "USDScene: failed to create mesh \"%s\": %s",
                    mi.prim_path.c_str(), e.what());
            }
        }

        Log(Info, "USDScene \"%s\": %zu meshes, %zu emitters (took %s)",
            file_path.filename().string(), mesh_count, m_emitters.size(),
            util::time_string((float) timer.value()));
    }

    std::vector<ref<Object>> expand() const override {
        // Return shapes + emitters
        std::vector<ref<Object>> result;
        result.reserve(m_shapes.size() + m_emitters.size());
        result.insert(result.end(), m_shapes.begin(), m_shapes.end());
        result.insert(result.end(), m_emitters.begin(), m_emitters.end());
        return result;
    }

    ScalarBoundingBox3f bbox() const override { return ScalarBoundingBox3f(); }

    MI_DECLARE_CLASS(USDScene)

private:
    using ScalarTransform4f = typename Shape<Float, Spectrum>::ScalarTransform4f;

    // --- Transform helpers ---

    static ScalarTransform4f to_mi_transform(const tinyusdz::value::matrix4d &m) {
        using Matrix4f = dr::Matrix<float, 4>;
        Matrix4f mat;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                mat(i, j) = (float) m.m[j][i]; // transpose (USD row-vec -> Mitsuba col-vec)
        return ScalarTransform4f(mat);
    }

    static tinyusdz::value::matrix4d get_local_xform(const tinyusdz::Prim &prim) {
        if (auto *mesh = prim.as<tinyusdz::GeomMesh>()) {
            auto result = mesh->GetLocalMatrix();
            if (result) return result.value();
        }
        if (auto *xform = prim.as<tinyusdz::Xform>()) {
            auto result = xform->GetLocalMatrix();
            if (result) return result.value();
        }
        if (auto *sl = prim.as<tinyusdz::SphereLight>()) {
            auto result = sl->GetLocalMatrix();
            if (result) return result.value();
        }
        if (auto *dl = prim.as<tinyusdz::DomeLight>()) {
            auto result = dl->GetLocalMatrix();
            if (result) return result.value();
        }
        if (auto *dist = prim.as<tinyusdz::DistantLight>()) {
            auto result = dist->GetLocalMatrix();
            if (result) return result.value();
        }
        if (auto *rl = prim.as<tinyusdz::RectLight>()) {
            auto result = rl->GetLocalMatrix();
            if (result) return result.value();
        }
        return tinyusdz::value::matrix4d::identity();
    }

    // --- MaterialX lookup ---

    const MtlxOpenPBRParams *lookup_mtlx(const std::string &mat_name) {
        if (mat_name.empty()) return nullptr;

        auto it = m_mtlx_cache.find(mat_name);
        if (it != m_mtlx_cache.end())
            return it->second.valid ? &it->second : nullptr;

        for (const auto &dir : {m_base_dir + "/materials", m_base_dir}) {
            std::string path = dir + "/" + mat_name + ".mtlx";
            if (fs::exists(path)) {
                MtlxOpenPBRParams params = parse_mtlx_openpbr(path, m_base_dir);
                auto result = m_mtlx_cache.insert({mat_name, params});
                if (params.valid) {
                    Log(Info, "USDScene: loaded material \"%s\" (tex: %s)",
                        mat_name.c_str(),
                        params.base_color_tex.empty() ? "none" : "yes");
                    return &result.first->second;
                }
                return nullptr;
            }
        }

        MtlxOpenPBRParams empty;
        m_mtlx_cache.insert({mat_name, empty});
        return nullptr;
    }

    static std::string material_name_from_path(const std::string &path) {
        auto pos = path.rfind('/');
        if (pos != std::string::npos)
            return path.substr(pos + 1);
        return path;
    }

    // --- Texture creation ---

    /// Create a Mitsuba bitmap texture from a file path.
    /// raw=true for non-color data (roughness, metalness, etc.)
    ref<Object> create_bitmap_texture(const std::string &path, bool raw) {
        if (!fs::exists(path)) {
            Log(Warn, "USDScene: texture not found: %s", path.c_str());
            return nullptr;
        }
        try {
            Properties tex_props;
            tex_props.set_plugin_name("bitmap");
            tex_props.set("filename", path);
            tex_props.set("raw", raw);
            auto obj = PluginManager::instance()->create_object(
                tex_props, Texture::Variant, ObjectType::Texture);

            // BitmapTexture is a factory — expand() returns the real impl
            auto expanded = obj->expand();
            if (!expanded.empty())
                obj = expanded[0];

            return obj;
        } catch (const std::exception &e) {
            Log(Warn, "USDScene: failed to load texture '%s': %s",
                path.c_str(), e.what());
            return nullptr;
        }
    }

    // --- BSDF creation ---

    ref<Object> create_openpbr_bsdf(const MtlxOpenPBRParams *mtlx,
                                     ScalarColor3f fallback_color) {
        // Clamp colors to [0,1] to avoid SRGBReflectanceSpectrum errors
        auto clamp01 = [](const std::array<float,3> &c) -> ScalarColor3f {
            return ScalarColor3f(std::min(std::max(c[0], 0.f), 1.f),
                                 std::min(std::max(c[1], 0.f), 1.f),
                                 std::min(std::max(c[2], 0.f), 1.f));
        };
        for (int i = 0; i < 3; ++i)
            fallback_color[i] = std::min(std::max(fallback_color[i], 0.f), 1.f);

        Properties bsdf_props;
        bsdf_props.set_plugin_name("openpbr");

        if (mtlx && mtlx->valid) {
            bsdf_props.set("base_weight", mtlx->base_weight);

            // base_color (texturable, sRGB)
            if (!mtlx->base_color_tex.empty()) {
                auto tex = create_bitmap_texture(mtlx->base_color_tex,
                                                  !mtlx->base_color_srgb);
                if (tex) bsdf_props.set("base_color", tex);
                else     bsdf_props.set("base_color", fallback_color);
            } else if (mtlx->has_base_color) {
                bsdf_props.set("base_color", clamp01(mtlx->base_color));
            } else {
                bsdf_props.set("base_color", fallback_color);
            }

            // base_metalness (texturable, raw)
            if (!mtlx->base_metalness_tex.empty()) {
                auto tex = create_bitmap_texture(mtlx->base_metalness_tex, true);
                if (tex) bsdf_props.set("base_metalness", tex);
                else if (mtlx->has_metalness)
                    bsdf_props.set("base_metalness", mtlx->base_metalness);
            } else if (mtlx->has_metalness) {
                bsdf_props.set("base_metalness", mtlx->base_metalness);
            }

            // specular
            if (mtlx->has_specular_weight)
                bsdf_props.set("specular_weight", mtlx->specular_weight);

            // specular_color (texturable, sRGB)
            if (!mtlx->specular_color_tex.empty()) {
                auto tex = create_bitmap_texture(mtlx->specular_color_tex, false);
                if (tex) bsdf_props.set("specular_color", tex);
                else if (mtlx->has_specular_color)
                    bsdf_props.set("specular_color", clamp01(mtlx->specular_color));
            } else if (mtlx->has_specular_color) {
                bsdf_props.set("specular_color", clamp01(mtlx->specular_color));
            }

            // specular_roughness (texturable, raw)
            if (!mtlx->specular_roughness_tex.empty()) {
                auto tex = create_bitmap_texture(mtlx->specular_roughness_tex, true);
                if (tex) bsdf_props.set("specular_roughness", tex);
                else if (mtlx->has_specular_roughness)
                    bsdf_props.set("specular_roughness", mtlx->specular_roughness);
            } else if (mtlx->has_specular_roughness) {
                bsdf_props.set("specular_roughness", mtlx->specular_roughness);
            }

            if (mtlx->has_specular_ior)
                bsdf_props.set("specular_ior", mtlx->specular_ior);

            // transmission_weight (texturable, raw)
            if (!mtlx->transmission_weight_tex.empty()) {
                auto tex = create_bitmap_texture(mtlx->transmission_weight_tex, true);
                if (tex) bsdf_props.set("transmission_weight", tex);
                else if (mtlx->has_transmission)
                    bsdf_props.set("transmission_weight", mtlx->transmission_weight);
            } else if (mtlx->has_transmission) {
                bsdf_props.set("transmission_weight", mtlx->transmission_weight);
            }

            // transmission_color (texturable, sRGB)
            if (!mtlx->transmission_color_tex.empty()) {
                auto tex = create_bitmap_texture(mtlx->transmission_color_tex, false);
                if (tex) bsdf_props.set("transmission_color", tex);
                else if (mtlx->has_transmission_color)
                    bsdf_props.set("transmission_color", clamp01(mtlx->transmission_color));
            } else if (mtlx->has_transmission_color) {
                bsdf_props.set("transmission_color", clamp01(mtlx->transmission_color));
            }

            // geometry_thin_walled
            if (mtlx->has_thin_walled)
                bsdf_props.set("geometry_thin_walled", mtlx->thin_walled);

            // coat_weight (texturable, raw)
            if (!mtlx->coat_weight_tex.empty()) {
                auto tex = create_bitmap_texture(mtlx->coat_weight_tex, true);
                if (tex) bsdf_props.set("coat_weight", tex);
                else if (mtlx->has_coat)
                    bsdf_props.set("coat_weight", mtlx->coat_weight);
            } else if (mtlx->has_coat) {
                bsdf_props.set("coat_weight", mtlx->coat_weight);
            }

            // coat_color (texturable, sRGB)
            if (!mtlx->coat_color_tex.empty()) {
                auto tex = create_bitmap_texture(mtlx->coat_color_tex, false);
                if (tex) bsdf_props.set("coat_color", tex);
                else if (mtlx->has_coat_color)
                    bsdf_props.set("coat_color", clamp01(mtlx->coat_color));
            } else if (mtlx->has_coat_color) {
                bsdf_props.set("coat_color", clamp01(mtlx->coat_color));
            }

            // coat_roughness (texturable, raw)
            if (!mtlx->coat_roughness_tex.empty()) {
                auto tex = create_bitmap_texture(mtlx->coat_roughness_tex, true);
                if (tex) bsdf_props.set("coat_roughness", tex);
                else if (mtlx->has_coat_roughness)
                    bsdf_props.set("coat_roughness", mtlx->coat_roughness);
            } else if (mtlx->has_coat_roughness) {
                bsdf_props.set("coat_roughness", mtlx->coat_roughness);
            }

            if (mtlx->has_coat_ior)
                bsdf_props.set("coat_ior", mtlx->coat_ior);
        } else {
            bsdf_props.set("base_color", fallback_color);
        }

        auto inner = PluginManager::instance()->create_object(
            bsdf_props, BSDF::Variant, ObjectType::BSDF);

        // Wrap in twosided (skip for transmission materials)
        bool has_transmission = mtlx && mtlx->has_transmission
                                && (mtlx->transmission_weight > 0.0f
                                    || !mtlx->transmission_weight_tex.empty());
        if (!has_transmission) {
            Properties ts_props;
            ts_props.set_plugin_name("twosided");
            ts_props.set("bsdf", inner);
            return PluginManager::instance()->create_object(
                ts_props, BSDF::Variant, ObjectType::BSDF);
        }
        return inner;
    }

    // --- Material color extraction ---

    static ScalarColor3f get_material_color(
        const tinyusdz::GeomMesh *mesh,
        const tinyusdz::Prim &prim,
        const tinyusdz::Stage &stage)
    {
        if (mesh->has_materialBinding()) {
            auto &rel = mesh->materialBinding.value();
            auto target = rel.targetPath;
            if (target.is_valid()) {
                const tinyusdz::Prim *mat_prim = nullptr;
                std::string err;
                if (stage.find_prim_at_path(
                        tinyusdz::Path(target.prim_part(), ""),
                        mat_prim, &err) && mat_prim) {
                    auto *material = mat_prim->as<tinyusdz::Material>();
                    if (material && material->surface.authored()) {
                        auto conns = material->surface.get_connections();
                        if (!conns.empty()) {
                            const tinyusdz::Prim *shader_prim = nullptr;
                            if (stage.find_prim_at_path(
                                    tinyusdz::Path(conns[0].prim_part(), ""),
                                    shader_prim, &err) && shader_prim) {
                                auto *shader = shader_prim->as<tinyusdz::Shader>();
                                if (shader) {
                                    auto *ps = shader->value.as<tinyusdz::UsdPreviewSurface>();
                                    if (ps) {
                                        tinyusdz::value::color3f c;
                                        if (ps->diffuseColor.get_value().get_scalar(&c))
                                            return ScalarColor3f(c[0], c[1], c[2]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        tinyusdz::value::color3f col;
        if (mesh->get_displayColor(&col))
            return ScalarColor3f(col[0], col[1], col[2]);

        auto colors = mesh->get_displayColors();
        if (!colors.empty())
            return ScalarColor3f(colors[0][0], colors[0][1], colors[0][2]);

        return ScalarColor3f(0.5f, 0.5f, 0.5f);
    }

    static std::string get_bound_material_name(
        const tinyusdz::GeomMesh *mesh,
        const tinyusdz::Prim &prim)
    {
        if (mesh->has_materialBinding()) {
            auto &rel = mesh->materialBinding.value();
            auto target = rel.targetPath;
            if (target.is_valid())
                return material_name_from_path(target.prim_part());
        }

        // Fallback: read userProperties:materialName custom attribute
        // (set by flatten_geo.py for flattened geometry files)
        auto it = mesh->props.find("userProperties:materialName");
        if (it != mesh->props.end()) {
            const auto &prop = it->second;
            if (prop.is_attribute()) {
                auto val = prop.get_attribute().get_value<std::string>();
                if (val)
                    return val.value();
            }
        }

        return {};
    }

    // --- Light info ---

    struct LightInfo {
        std::string name;
        std::string type;   // "sphere", "dome", "distant", "rect", "disk"
        tinyusdz::value::matrix4d world_matrix;

        // Common light properties
        std::array<float,3> color = {1.f, 1.f, 1.f};
        float intensity = 1.0f;
        float exposure = 0.0f;

        // SphereLight
        float radius = 0.5f;

        // DomeLight
        std::string texture_file;

        // DistantLight
        float angle = 0.53f;

        // RectLight
        float width = 2.0f;
        float height = 2.0f;
    };

    /// Helper to get light color
    template <typename LightT>
    static void extract_common_light(const LightT *light,
                                      const std::string &prim_name,
                                      LightInfo &li) {
        li.name = prim_name;
        // Color
        {
            tinyusdz::value::color3f c;
            if (light->color.get_value().get_scalar(&c)) {
                li.color = {c[0], c[1], c[2]};
            }
        }
        // Intensity
        {
            float v;
            if (light->intensity.get_value().get_scalar(&v))
                li.intensity = v;
        }
        // Exposure
        {
            float v;
            if (light->exposure.get_value().get_scalar(&v))
                li.exposure = v;
        }
    }

    // --- Prim traversal ---

    struct MeshInfo {
        std::string prim_path;
        tinyusdz::value::matrix4d world_matrix;
        std::string mat_name;
        ScalarColor3f color;
    };

    void collect_prims(const tinyusdz::Prim &prim,
                       const tinyusdz::value::matrix4d &parent_world,
                       const tinyusdz::Stage &stage,
                       std::vector<MeshInfo> &meshes,
                       std::vector<LightInfo> &lights) {
        auto local = get_local_xform(prim);
        auto world = local * parent_world;

        // Mesh
        if (auto *mesh = prim.as<tinyusdz::GeomMesh>()) {
            auto points = mesh->get_points();
            if (!points.empty()) {
                MeshInfo mi;
                mi.prim_path = prim.absolute_path().full_path_name();
                mi.world_matrix = world;
                mi.mat_name = get_bound_material_name(mesh, prim);
                mi.color = get_material_color(mesh, prim, stage);
                meshes.push_back(std::move(mi));
            }
        }

        // SphereLight
        if (auto *sl = prim.as<tinyusdz::SphereLight>()) {
            LightInfo li;
            li.type = "sphere";
            extract_common_light(sl, prim.element_path().prim_part(), li);
            li.world_matrix = world;
            float r;
            if (sl->radius.get_value().get_scalar(&r))
                li.radius = r;
            lights.push_back(li);
        }

        // DomeLight
        if (auto *dl = prim.as<tinyusdz::DomeLight>()) {
            LightInfo li;
            li.type = "dome";
            extract_common_light(dl, prim.element_path().prim_part(), li);
            li.world_matrix = world;
            // Texture file (TypedAttribute without fallback)
            auto file_opt = dl->file.get_value();
            if (file_opt.has_value()) {
                tinyusdz::value::AssetPath ap;
                if (file_opt.value().get_scalar(&ap)) {
                    std::string tex = ap.GetAssetPath();
                    if (!tex.empty()) {
                        // Try scene dir, then textures subdir
                        li.texture_file = m_base_dir + "/" + tex;
                        if (!fs::exists(li.texture_file)) {
                            auto slash = tex.rfind('/');
                            std::string fname = (slash != std::string::npos)
                                ? tex.substr(slash + 1) : tex;
                            li.texture_file = m_base_dir + "/textures/" + fname;
                        }
                        if (fs::exists(li.texture_file))
                            li.texture_file = std::filesystem::canonical(li.texture_file).string();
                        else
                            li.texture_file.clear();
                    }
                }
            }
            // Only add if we have a texture; otherwise user defines envmap in XML
            if (!li.texture_file.empty())
                lights.push_back(li);
            else
                Log(Warn, "USDScene: DomeLight '%s' has no resolved texture "
                    "(define envmap in scene XML)", li.name.c_str());
        }

        // DistantLight
        if (auto *distant = prim.as<tinyusdz::DistantLight>()) {
            LightInfo li;
            li.type = "distant";
            extract_common_light(distant, prim.element_path().prim_part(), li);
            li.world_matrix = world;
            float a;
            if (distant->angle.get_value().get_scalar(&a))
                li.angle = a;
            lights.push_back(li);
        }

        // RectLight
        if (auto *rl = prim.as<tinyusdz::RectLight>()) {
            LightInfo li;
            li.type = "rect";
            extract_common_light(rl, prim.element_path().prim_part(), li);
            li.world_matrix = world;
            float w, h;
            if (rl->width.get_value().get_scalar(&w))
                li.width = w;
            if (rl->height.get_value().get_scalar(&h))
                li.height = h;
            lights.push_back(li);
        }

        for (const auto &child : prim.children())
            collect_prims(child, world, stage, meshes, lights);
    }

    // --- Emitter creation ---

    ref<Object> create_emitter(const LightInfo &li) {
        // Compute effective radiance multiplier: intensity * 2^exposure
        float power = li.intensity * std::pow(2.0f, li.exposure);
        ScalarColor3f radiance(li.color[0] * power,
                                li.color[1] * power,
                                li.color[2] * power);

        auto make = [&](Properties &em_props) -> ref<Object> {
            return PluginManager::instance()->create_object(
                em_props, Emitter::Variant, ObjectType::Emitter);
        };

        if (li.type == "sphere") {
            Properties em_props;
            em_props.set_plugin_name("point");
            em_props.set("intensity", radiance);
            auto xform = to_mi_transform(li.world_matrix);
            em_props.set("to_world", xform);
            auto pos = xform * ScalarPoint3f(0.f, 0.f, 0.f);
            Log(Info, "USDScene: creating point light \"%s\" at (%.1f, %.1f, %.1f) "
                "intensity=(%.2f, %.2f, %.2f)",
                li.name.c_str(), pos.x(), pos.y(), pos.z(),
                radiance.r(), radiance.g(), radiance.b());
            return make(em_props);
        }

        if (li.type == "dome") {
            if (!li.texture_file.empty() && fs::exists(li.texture_file)) {
                Properties em_props;
                em_props.set_plugin_name("envmap");
                em_props.set("filename", li.texture_file);
                em_props.set("scale", power);
                em_props.set("to_world", to_mi_transform(li.world_matrix));
                Log(Info, "USDScene: creating envmap light \"%s\" from \"%s\"",
                    li.name.c_str(), li.texture_file.c_str());
                return make(em_props);
            } else {
                Properties em_props;
                em_props.set_plugin_name("constant");
                em_props.set("radiance", radiance);
                Log(Info, "USDScene: creating constant env light \"%s\" "
                    "radiance=(%.3f, %.3f, %.3f)",
                    li.name.c_str(),
                    radiance.r(), radiance.g(), radiance.b());
                return make(em_props);
            }
        }

        if (li.type == "distant") {
            Properties em_props;
            em_props.set_plugin_name("directional");
            em_props.set("irradiance", radiance);
            em_props.set("to_world", to_mi_transform(li.world_matrix));
            Log(Info, "USDScene: creating directional light \"%s\"",
                li.name.c_str());
            return make(em_props);
        }

        if (li.type == "rect") {
            // RectLight → rectangle shape with area emitter
            Properties shape_props;
            shape_props.set_plugin_name("rectangle");
            auto xform = to_mi_transform(li.world_matrix);
            // Scale rectangle to match USD width/height (Mitsuba rect is 2x2)
            using Matrix4f = dr::Matrix<float, 4>;
            Matrix4f scale_mat;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    scale_mat(i, j) = (i == j) ? 1.f : 0.f;
            scale_mat(0, 0) = li.width * 0.5f;
            scale_mat(1, 1) = li.height * 0.5f;
            auto scaled_xform = xform * ScalarTransform4f(scale_mat);
            shape_props.set("to_world", scaled_xform);

            // Area emitter
            Properties em_props;
            em_props.set_plugin_name("area");
            em_props.set("radiance", radiance);
            auto emitter = PluginManager::instance()->create_object(
                em_props, Emitter::Variant, ObjectType::Emitter);
            shape_props.set("emitter", emitter);

            auto pos = xform * ScalarPoint3f(0.f, 0.f, 0.f);
            Log(Info, "USDScene: creating rect light \"%s\" at (%.1f, %.1f, %.1f) "
                "size=%.1fx%.1f radiance=(%.2f, %.2f, %.2f)",
                li.name.c_str(), pos.x(), pos.y(), pos.z(),
                li.width, li.height,
                radiance.r(), radiance.g(), radiance.b());

            // Return as a shape (not emitter) — it goes to m_shapes
            auto shape = PluginManager::instance()->create_object(
                shape_props, Shape::Variant, ObjectType::Shape);
            return shape;
        }

        Log(Warn, "USDScene: unsupported light type \"%s\" for \"%s\"",
            li.type.c_str(), li.name.c_str());
        return nullptr;
    }

    std::vector<ref<Object>> m_shapes;
    std::vector<ref<Object>> m_emitters;
    std::string m_base_dir;
    std::unordered_map<std::string, MtlxOpenPBRParams> m_mtlx_cache;
};

MI_EXPORT_PLUGIN(USDScene)
NAMESPACE_END(mitsuba)
