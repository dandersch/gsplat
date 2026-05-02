#include "hotspot.h"
#include "refview.h"
#include "json_mini.h"
#include <SDL3/SDL.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>

// ---------------------------------------------------------------------------
// On-disk format (.hotspots sidecar, JSON):
//
// {
//   "version": 1,
//   "image":   "n01w02_equirect.png",      // optional sanity check
//   "hotspots": [
//     {
//       "shape":  { "type": "polygon",
//                   "points": [[u,v], [u,v], ...] },     // >= 3 points, [0,1]
//       "action": { "type": "warp", "target": "n01w03_equirect.png" }
//     }
//   ]
// }
//
// Polygons are implicitly closed (last -> first). Winding is irrelevant
// (hit test is even-odd). Seam-crossing regions are emitted by the
// converter as multiple independent hotspots sharing the same action.
// ---------------------------------------------------------------------------

// ---- internal types -------------------------------------------------------

// Mutable in-progress hotspot used during parsing. We don't know the polygon
// point count up front, so we grow dynamically and shrink to fit at the end.
struct HotspotBuild {
    HotspotShapeType  type;
    float           (*points)[2];
    uint32_t          point_count;
    uint32_t          point_cap;

    HotspotActionType action_type;
    char              action_target[256];   // image_name to resolve later
    bool              valid;                // false -> drop after parsing
};

static void build_free(HotspotBuild* b) {
    free(b->points);
    b->points = NULL;
    b->point_count = b->point_cap = 0;
}

static void build_push_point(HotspotBuild* b, float u, float v) {
    if (b->point_count == b->point_cap) {
        uint32_t newcap = b->point_cap ? b->point_cap * 2 : 8;
        b->points = (float(*)[2])realloc(b->points, newcap * sizeof(float[2]));
        b->point_cap = newcap;
    }
    b->points[b->point_count][0] = u;
    b->points[b->point_count][1] = v;
    b->point_count++;
}

// ---- parsing --------------------------------------------------------------

static bool parse_polygon_points(Json* j, HotspotBuild* b) {
    if (!json_expect_char(j, '[')) return false;
    if (json_try_char(j, ']')) return true;
    do {
        if (!json_expect_char(j, '[')) return false;
        float u, v;
        if (!json_parse_float(j, &u)) return false;
        if (!json_expect_char(j, ',')) return false;
        if (!json_parse_float(j, &v)) return false;
        if (!json_expect_char(j, ']')) return false;
        if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) {
            // Mark hotspot invalid but keep parsing the rest of the file.
            b->valid = false;
        }
        build_push_point(b, u, v);
    } while (json_try_char(j, ','));
    return json_expect_char(j, ']');
}

static bool parse_shape(Json* j, HotspotBuild* b) {
    if (!json_expect_char(j, '{')) return false;
    bool has_type = false;
    bool has_points = false;
    if (!json_try_char(j, '}')) {
        do {
            char key[64];
            if (!json_parse_string(j, key, sizeof(key))) return false;
            if (!json_expect_char(j, ':')) return false;
            if (strcmp(key, "type") == 0) {
                char val[32];
                if (!json_parse_string(j, val, sizeof(val))) return false;
                if (strcmp(val, "polygon") == 0) {
                    b->type = HOTSPOT_SHAPE_POLYGON;
                    has_type = true;
                } else {
                    SDL_Log("Hotspot: unknown shape.type \"%s\", dropping hotspot", val);
                    b->valid = false;
                    has_type = true; // pretend, so we don't double-log
                }
            } else if (strcmp(key, "points") == 0) {
                if (!parse_polygon_points(j, b)) return false;
                has_points = true;
            } else {
                json_skip_value(j);
            }
            if (!j->ok) return false;
        } while (json_try_char(j, ','));
        if (!json_expect_char(j, '}')) return false;
    }
    if (!has_type) { SDL_Log("Hotspot: shape missing 'type'"); b->valid = false; }
    if (b->type == HOTSPOT_SHAPE_POLYGON) {
        if (!has_points) { SDL_Log("Hotspot: polygon missing 'points'"); b->valid = false; }
        if (b->point_count < 3) {
            if (b->valid) SDL_Log("Hotspot: polygon has %u points (need >= 3), dropping", b->point_count);
            b->valid = false;
        }
    }
    return true;
}

static bool parse_action(Json* j, HotspotBuild* b) {
    if (!json_expect_char(j, '{')) return false;
    bool has_type = false;
    bool has_target = false;
    if (!json_try_char(j, '}')) {
        do {
            char key[64];
            if (!json_parse_string(j, key, sizeof(key))) return false;
            if (!json_expect_char(j, ':')) return false;
            if (strcmp(key, "type") == 0) {
                char val[32];
                if (!json_parse_string(j, val, sizeof(val))) return false;
                if (strcmp(val, "warp") == 0) {
                    b->action_type = HOTSPOT_ACTION_WARP;
                    has_type = true;
                } else {
                    SDL_Log("Hotspot: unknown action.type \"%s\", dropping hotspot", val);
                    b->valid = false;
                    has_type = true;
                }
            } else if (strcmp(key, "target") == 0) {
                if (!json_parse_string(j, b->action_target, sizeof(b->action_target))) return false;
                has_target = true;
            } else {
                json_skip_value(j);
            }
            if (!j->ok) return false;
        } while (json_try_char(j, ','));
        if (!json_expect_char(j, '}')) return false;
    }
    if (!has_type) { SDL_Log("Hotspot: action missing 'type'"); b->valid = false; }
    if (b->action_type == HOTSPOT_ACTION_WARP && !has_target) {
        SDL_Log("Hotspot: warp action missing 'target'"); b->valid = false;
    }
    return true;
}

static bool parse_hotspot(Json* j, HotspotBuild* b) {
    *b = {};
    b->valid = true;
    if (!json_expect_char(j, '{')) return false;
    bool has_shape = false;
    bool has_action = false;
    if (!json_try_char(j, '}')) {
        do {
            char key[64];
            if (!json_parse_string(j, key, sizeof(key))) return false;
            if (!json_expect_char(j, ':')) return false;
            if (strcmp(key, "shape") == 0) {
                if (!parse_shape(j, b)) return false;
                has_shape = true;
            } else if (strcmp(key, "action") == 0) {
                if (!parse_action(j, b)) return false;
                has_action = true;
            } else {
                json_skip_value(j);
            }
            if (!j->ok) return false;
        } while (json_try_char(j, ','));
        if (!json_expect_char(j, '}')) return false;
    }
    if (!has_shape || !has_action) {
        SDL_Log("Hotspot: entry missing shape or action"); b->valid = false;
    }
    return true;
}

// Parse a full sidecar buffer. On success, returns a heap-allocated array of
// HotspotBuild (caller takes ownership). The `image` field (if present) is
// copied into image_field_out for sanity check by the loader.
static bool parse_sidecar(const char* buf, size_t len,
                          HotspotBuild** out_builds, uint32_t* out_count,
                          char* image_field_out, size_t image_field_size,
                          const char* sidecar_path)
{
    *out_builds = NULL;
    *out_count = 0;
    if (image_field_size) image_field_out[0] = '\0';

    HotspotBuild* builds = NULL;
    uint32_t bcount = 0;
    uint32_t bcap = 0;
    int version = 0;
    bool have_version = false;

    Json j;
    json_init(&j, buf, len);

    if (!json_expect_char(&j, '{')) goto fail;

    if (!json_try_char(&j, '}')) {
        do {
            char key[64];
            if (!json_parse_string(&j, key, sizeof(key))) goto fail;
            if (!json_expect_char(&j, ':')) goto fail;

            if (strcmp(key, "version") == 0) {
                if (!json_parse_int(&j, &version)) goto fail;
                have_version = true;
            } else if (strcmp(key, "image") == 0) {
                json_parse_string(&j, image_field_out, image_field_size);
                if (!j.ok) goto fail;
            } else if (strcmp(key, "hotspots") == 0) {
                if (!json_expect_char(&j, '[')) goto fail;
                if (!json_try_char(&j, ']')) {
                    do {
                        if (bcount == bcap) {
                            bcap = bcap ? bcap * 2 : 8;
                            builds = (HotspotBuild*)realloc(builds, bcap * sizeof(HotspotBuild));
                        }
                        if (!parse_hotspot(&j, &builds[bcount])) goto fail;
                        bcount++;
                    } while (json_try_char(&j, ','));
                    if (!json_expect_char(&j, ']')) goto fail;
                }
            } else {
                json_skip_value(&j);
                if (!j.ok) goto fail;
            }
        } while (json_try_char(&j, ','));
        if (!json_expect_char(&j, '}')) goto fail;
    }

    if (!have_version) {
        SDL_Log("Hotspot [%s]: missing 'version' field", sidecar_path);
        goto fail;
    }
    if (version != 1) {
        SDL_Log("Hotspot [%s]: unsupported version %d (only 1 supported)", sidecar_path, version);
        goto fail;
    }

    *out_builds = builds;
    *out_count = bcount;
    return true;

fail:
    if (!j.ok) {
        SDL_Log("Hotspot [%s]: parse error at byte %d: %s",
                sidecar_path, j.err_offset, j.err_msg ? j.err_msg : "?");
    }
    if (builds) {
        for (uint32_t i = 0; i < bcount; i++) build_free(&builds[i]);
        free(builds);
    }
    return false;
}

// ---- path helpers ---------------------------------------------------------

// dir + basename(image_name without extension) + ".hotspots"
static void make_sidecar_path(const char* dir, const char* image_name,
                              char* out, size_t out_size)
{
    const char* slash_a = strrchr(image_name, '/');
    const char* slash_b = strrchr(image_name, '\\');
    const char* slash = slash_a > slash_b ? slash_a : slash_b;
    const char* dot = strrchr(image_name, '.');
    if (dot && (slash == NULL || dot > slash)) {
        snprintf(out, out_size, "%s/%.*s.hotspots", dir,
                 (int)(dot - image_name), image_name);
    } else {
        snprintf(out, out_size, "%s/%s.hotspots", dir, image_name);
    }
}

// basename of a path, e.g. "foo/bar.png" -> "bar.png"
static const char* path_basename(const char* p) {
    const char* a = strrchr(p, '/');
    const char* b = strrchr(p, '\\');
    const char* s = a > b ? a : b;
    return s ? s + 1 : p;
}

// ---- target resolution ----------------------------------------------------

static int32_t resolve_target(const RefViewSet* set, const char* name) {
    for (uint32_t i = 0; i < set->count; i++) {
        if (strcmp(set->views[i].image_name, name) == 0) return (int32_t)i;
    }
    return -1;
}

// ---- public API -----------------------------------------------------------

void hotspot_free_array(Hotspot* hotspots, uint32_t count) {
    if (!hotspots) return;
    for (uint32_t i = 0; i < count; i++) {
        if (hotspots[i].type == HOTSPOT_SHAPE_POLYGON) {
            free(hotspots[i].polygon.points);
        }
    }
    free(hotspots);
}

static void load_one_view(RefViewSet* set, RefView* v) {
    char sidecar[1024];
    make_sidecar_path(set->image_dir, v->image_name, sidecar, sizeof(sidecar));

    size_t data_size = 0;
    void* data = SDL_LoadFile(sidecar, &data_size);
    if (!data) {
        // Missing file is silent (most views won't have hotspots).
        // Distinguish "not found" from real I/O errors? SDL_LoadFile sets an
        // error message either way; we just skip.
        return;
    }

    HotspotBuild* builds = NULL;
    uint32_t bcount = 0;
    char image_field[256] = {};

    bool ok = parse_sidecar((const char*)data, data_size,
                            &builds, &bcount,
                            image_field, sizeof(image_field),
                            sidecar);
    SDL_free(data);
    if (!ok) return;

    // Sanity-check `image` field if present.
    if (image_field[0]) {
        const char* my_base = path_basename(v->image_name);
        if (strcmp(image_field, my_base) != 0 &&
            strcmp(image_field, v->image_name) != 0) {
            SDL_Log("Hotspot [%s]: image field \"%s\" does not match view \"%s\", skipping",
                    sidecar, image_field, v->image_name);
            for (uint32_t i = 0; i < bcount; i++) build_free(&builds[i]);
            free(builds);
            return;
        }
    }

    // Resolve targets and produce final Hotspot array, dropping invalid entries.
    Hotspot* finals = (Hotspot*)calloc(bcount, sizeof(Hotspot));
    uint32_t fcount = 0;
    for (uint32_t i = 0; i < bcount; i++) {
        HotspotBuild* b = &builds[i];
        if (!b->valid) { build_free(b); continue; }
        if (b->action_type == HOTSPOT_ACTION_WARP) {
            int32_t target = resolve_target(set, b->action_target);
            if (target < 0) {
                SDL_Log("Hotspot [%s]: warp target \"%s\" not found, dropping hotspot",
                        sidecar, b->action_target);
                build_free(b);
                continue;
            }
            // Move polygon ownership into final.
            Hotspot* h = &finals[fcount++];
            h->type = b->type;
            h->polygon.points = b->points;
            h->polygon.count  = b->point_count;
            h->action.type = HOTSPOT_ACTION_WARP;
            h->action.warp.target_view = target;
            // Detach from build so build_free won't double-free.
            b->points = NULL;
            b->point_count = b->point_cap = 0;
        } else {
            build_free(b);
        }
    }
    free(builds);

    if (fcount == 0) {
        free(finals);
        SDL_Log("Hotspot [%s]: no valid hotspots after parse", sidecar);
        return;
    }

    // Shrink-to-fit (optional; tiny arrays).
    if (fcount < bcount) {
        Hotspot* shrunk = (Hotspot*)realloc(finals, fcount * sizeof(Hotspot));
        if (shrunk) finals = shrunk;
    }

    v->hotspots = finals;
    v->hotspot_count = fcount;
    SDL_Log("Hotspot [%s]: loaded %u hotspot(s)", sidecar, fcount);
}

void hotspot_load_for_set(RefViewSet* set) {
    if (!set || !set->views) return;
    uint32_t total = 0;
    for (uint32_t i = 0; i < set->count; i++) {
        load_one_view(set, &set->views[i]);
        total += set->views[i].hotspot_count;
    }
    SDL_Log("Hotspot: %u hotspot(s) across %u view(s)", total, set->count);
}

// Standard even-odd / ray-cast point-in-polygon test.
static bool point_in_polygon(float u, float v, const float (*pts)[2], uint32_t n) {
    bool inside = false;
    for (uint32_t i = 0, j = n - 1; i < n; j = i++) {
        float ui = pts[i][0], vi = pts[i][1];
        float uj = pts[j][0], vj = pts[j][1];
        if ((vi > v) != (vj > v)) {
            float x_at = (uj - ui) * (v - vi) / (vj - vi) + ui;
            if (u < x_at) inside = !inside;
        }
    }
    return inside;
}

int32_t hotspot_pick(const RefView* view, float u, float v) {
    if (!view || !view->hotspots || view->hotspot_count == 0) return -1;
    for (uint32_t i = 0; i < view->hotspot_count; i++) {
        const Hotspot* h = &view->hotspots[i];
        if (h->type == HOTSPOT_SHAPE_POLYGON) {
            if (point_in_polygon(u, v, h->polygon.points, h->polygon.count)) {
                return (int32_t)i;
            }
        }
    }
    return -1;
}
