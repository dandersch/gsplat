#pragma once
#include <cstdint>

// Hotspot data attached to a RefView's panorama image, loaded from a sidecar
// file `<basename(image_name)>.hotspots` next to the image. See spec/comments
// in hotspot.cpp for the on-disk format.

enum HotspotShapeType : uint8_t {
    HOTSPOT_SHAPE_POLYGON = 0,
};

enum HotspotActionType : uint8_t {
    HOTSPOT_ACTION_WARP    = 0,  // lerp to another refview's panorama
    HOTSPOT_ACTION_INSPECT = 1,  // lerp to a free camera transform + go orthographic
};

struct HotspotPolygon {
    float    (*points)[2];   // owned, `count` entries; UVs in [0,1]
    uint32_t   count;
};

struct HotspotActionWarp {
    int32_t target_view;     // resolved RefView index, or -1 if unresolved (drop on load)
};

struct HotspotActionInspect {
    float position[3];       // world-space camera position
    float yaw;               // radians
    float pitch;             // radians
};

struct HotspotAction {
    HotspotActionType    type;
    HotspotActionWarp    warp;     // active iff type == HOTSPOT_ACTION_WARP
    HotspotActionInspect inspect;  // active iff type == HOTSPOT_ACTION_INSPECT
};

struct Hotspot {
    HotspotShapeType type;
    HotspotPolygon   polygon;  // active iff type == HOTSPOT_SHAPE_POLYGON
    HotspotAction    action;
};

struct RefView;     // fwd
struct RefViewSet;  // fwd

// Load a sidecar for every view in `set`. Missing sidecars are silent;
// parse errors / schema violations log a warning and produce zero hotspots
// for the affected view. Targets are resolved to internal indices here.
void hotspot_load_for_set(RefViewSet* set);

// Free the hotspots array attached to a single view (frees per-hotspot
// owned memory plus the array itself). Safe on (NULL, 0).
void hotspot_free_array(Hotspot* hotspots, uint32_t count);

// Hit test. Returns the index of the topmost (first-listed) hotspot
// containing (u, v) in [0,1] equirect UV space, or -1 if none.
int32_t hotspot_pick(const RefView* view, float u, float v);
