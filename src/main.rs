use bevy::{
    color::palettes::basic::PURPLE,
    prelude::*,
    sprite::MaterialMesh2dBundle,
    window::{WindowMode, WindowResolution},
};
use bevy_internal::{
    //diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    input::common_conditions::input_toggle_active,
    //sprite::Mesh2dHandle,
    window::PresentMode,
};

use bevy::window::Window;

use bevy_inspector_egui::{
    bevy_egui::EguiPlugin, prelude::ReflectInspectorOptions, quick::WorldInspectorPlugin,
    DefaultInspectorConfigPlugin, InspectorOptions,
};

use chrono::prelude::Utc;

mod math;
use math::*;

mod spatial_hash;
use spatial_hash::*;

//use bevy_hanabi::Gradient;

mod ui;
use systems::{
    bounce_system, calculate_density_system, cccb_display_system, gravity_system,
    keyboard_interaction_system, measurements_system, mouse_interaction_system, move_system,
    pressure_force_system, touch_interaction_system, update_spatial_hash_system,
};
use ui::*;

mod file_io;
use file_io::*;

mod utils;
use utils::*;

//mod colors;

mod systems;

/*
#[derive(Resource)]
struct GradientResource {
    gradient: Gradient<Vec4>,
    precomputed_materials: Vec<Handle<ColorMaterial>>,
}


#[derive(Resource)]
struct ColorSchemeCategoricalResource {
    colors: Vec<Color>,
    precomputed_materials: Vec<Handle<ColorMaterial>>,
}
*/

#[derive(Reflect, Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq)]
enum ParticleColorMode {
    Velocity,
    Density,
    CellKey,
    Blue,
}

fn default_particle_color_mode() -> ParticleColorMode {
    ParticleColorMode::Velocity
}

#[derive(Resource, Default, Clone)]
pub struct Measurements {
    delta_t: f32,
    tps: f32,
    p0_position: Vec3,
    p0_predicted_position: Vec3,
    p0_velocity: Vec2,
    p0_density: Density,
    p0_max_density_far: f32,
}

#[derive(Default, Reflect, Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct BoundingBox {
    width: f32,
    height: f32,
}

fn default_bounding_box() -> BoundingBox {
    BoundingBox {
        width: 16.,
        height: 9.,
    }
}

#[derive(Component)]
struct Particle;

#[derive(Component, Clone, Debug)]
struct Velocity(Vec2);

fn default_max_velocity_for_color() -> f32 {
    3.0
}

#[derive(Component, Clone, Debug, Default, Copy)]
struct Density {
    far: f32,
    near: f32,
}

fn default_target_density() -> f32 {
    36.
}

fn default_max_density_for_color() -> f32 {
    default_target_density() * 1.5
}

#[derive(Component, Clone, Debug)]
struct PredictedPosition(Vec3);

#[derive(Clone, Debug)]
struct SpatialIndex {
    key: u32,
    hash: u32,
    entity_id: Entity,
}

impl Default for SpatialIndex {
    fn default() -> Self {
        Self {
            key: u32::MAX,
            hash: u32::MAX,
            entity_id: Entity::from_raw(0),
        }
    }
}

#[derive(Resource)]
struct SpatialHash {
    indices: Vec<SpatialIndex>,
    offsets: Vec<usize>,
    first_entity_id: Entity,
}

fn get_default_interaction_input_strength() -> f32 {
    30.
}

fn get_default_interaction_input_radius() -> f32 {
    2.
}

fn get_default_time_scale() -> f32 {
    1.
}

fn get_default_prediction_time_scale() -> f32 {
    0.5
}

#[derive(
    Resource, Reflect, InspectorOptions, serde::Serialize, serde::Deserialize, Debug, Clone, Copy,
)]
#[reflect(Resource, InspectorOptions)]
pub struct Config {
    #[inspector(min = 0, max = 5000, speed = 1.)]
    num_particles: usize,
    gravity: Vec2,
    #[inspector(min = 0.0, max = 1.0, speed = 0.01)]
    damping: f32,
    #[inspector(min = 0.0, speed = 0.1)]
    target_density: f32,
    #[inspector(min = 0.0, speed = 0.1)]
    pressure_multiplier: f32,
    #[inspector(min = 0.0, speed = 0.1)]
    #[serde(default)]
    near_pressure_multiplier: f32,
    #[inspector(min = 0.0000001, max = 1000.0, speed = 0.005)]
    smoothing_radius: f32,
    #[inspector(min = 0.01, max = 10000.0, speed = 0.1)]
    #[serde(default = "default_max_velocity_for_color")]
    max_velocity_for_color: f32,
    #[inspector(min = 0.01, speed = 0.1)]
    #[serde(default = "default_max_density_for_color")]
    max_density_for_color: f32,
    #[serde(default = "default_particle_color_mode")]
    particle_color_mode: ParticleColorMode,
    #[serde(default)]
    mark_sample_particle_neighbors: bool,
    #[serde(default = "default_bounding_box")]
    bounding_box: BoundingBox,
    #[inspector(min = 0.0)]
    #[serde(default = "get_default_interaction_input_strength")]
    interaction_input_strength: f32,
    #[inspector(min = 0.0)]
    #[serde(default = "get_default_interaction_input_radius")]
    interaction_input_radius: f32,
    #[inspector(min = 0.0, max = 5.0, speed = 0.01)]
    #[serde(default = "get_default_time_scale")]
    time_scale: f32,
    #[inspector(min = 0.0, max = 6.0, speed = 0.01)]
    #[serde(default = "get_default_prediction_time_scale")]
    prediction_time_scale: f32,
    is_paused: bool,
    #[serde(default)]
    pause_after_next_frame: bool,
    start_time: i64,
    auto_save: bool,
    #[serde(default)]
    cccb_display_sender: bool,
}

const MASS: f32 = 1.;
const TIME_STEP: f64 = 1. / 180.;

// macos
// native resolution 2560-by-1664 pixels at 224 ppi
// measured: 3840 x 2496 => 1.5
static NATIVE_MULTIPLIER: f32 = 1.5;

// 1 for fullscreen
// 0.5 for twice horizontal aspect
static ASPECT_MULTIPLIER_Y: f32 = 0.55;

static SCREEN_PIXELS_X: f32 = 2560. * NATIVE_MULTIPLIER; // 5120
static SCREEN_PIXELS_Y: f32 = 1664. * NATIVE_MULTIPLIER * ASPECT_MULTIPLIER_Y; // 3328 (1331)

static SCALE_FACTOR: f32 = SCALE_FACTOR2 * 6.4 * 0.001; // 0.015;

static SCALE_FACTOR2: f32 = 0.5; // bigger makes things smaller

//pub const SCALE_FACTOR: f32 = 0.015;
//pub const SCALE_FACTOR: f32 = if cfg!(target_arch = "wasm32") {
//    0.015
//} else {
//    0.02
//};
static CIRCLE_RATIO: f32 = 0.11;

impl Default for Config {
    fn default() -> Self {
        Self {
            gravity: Vec2::new(0., -9.),
            damping: 0.2,
            target_density: default_target_density(),
            pressure_multiplier: 370.,
            near_pressure_multiplier: 7.2,
            smoothing_radius: 0.35,
            max_velocity_for_color: default_max_velocity_for_color(),
            max_density_for_color: default_max_density_for_color(),
            num_particles: 1200,
            particle_color_mode: default_particle_color_mode(),
            mark_sample_particle_neighbors: false,
            bounding_box: default_bounding_box(),
            interaction_input_strength: get_default_interaction_input_strength(),
            interaction_input_radius: 2.,
            time_scale: 1.,
            prediction_time_scale: 0.5,
            is_paused: false,
            pause_after_next_frame: false,
            start_time: Utc::now().timestamp(),
            auto_save: false,
            cccb_display_sender: false,
        }
    }
}

#[derive(Resource)]
struct InteractionInputs {
    point: Option<Vec2>,
    strength: f32,
}

fn main() {
    let config = load_most_recent_config_from_file();

    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "🌊".into(),
                    present_mode: PresentMode::AutoNoVsync,
                    mode: WindowMode::Windowed,
                    resolution: WindowResolution::new(SCREEN_PIXELS_X, SCREEN_PIXELS_Y)
                        .with_scale_factor_override(1.0),
                    ..default()
                }),
                ..default()
            }),
            //FrameTimeDiagnosticsPlugin,
            //LogDiagnosticsPlugin::default(),
        ))
        .add_plugins(EguiPlugin)
        .add_plugins(DefaultInspectorConfigPlugin)
        .add_plugins(
            WorldInspectorPlugin::default().run_if(input_toggle_active(false, KeyCode::Escape)),
        )
        .insert_resource(ClearColor(Color::srgb(0.0, 0.0, 0.0)))
        .insert_resource(config)
        .register_type::<Config>()
        //.insert_resource(GradientResource::new())
        //.insert_resource(ColorSchemeCategoricalResource::new())
        .insert_resource(SpatialHash {
            indices: Vec::<SpatialIndex>::new(),
            offsets: Vec::new(),
            first_entity_id: Entity::from_raw(0),
        })
        .insert_resource(Measurements::default())
        .insert_resource(InteractionInputs {
            point: None,
            strength: 0.,
        })
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                inspector_ui,
                keyboard_interaction_system,
                touch_interaction_system,
                mouse_interaction_system,
                cccb_display_system,
            ),
        )
        .add_systems(
            FixedUpdate,
            (
                gravity_system,
                update_spatial_hash_system,
                calculate_density_system,
                measurements_system,
                pressure_force_system,
                move_system,
                //sync_meshes_system,
                bounce_system,
                //color_system,
            )
                .chain(),
        )
        .insert_resource(Time::<Fixed>::from_seconds(TIME_STEP))
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    //mut gradient_resource: ResMut<GradientResource>,
    //mut color_scheme_categorical_resource: ResMut<ColorSchemeCategoricalResource>,
    config: Res<Config>,
) {
    commands.spawn(Camera2dBundle {
        projection: OrthographicProjection {
            scale: SCALE_FACTOR,
            far: 100.,
            near: -10.,
            ..default()
        },

        ..default()
    });

    //gradient_resource.precompute_materials(&mut materials);
    //color_scheme_categorical_resource.precompute_materials(&mut materials);

    // spawn particles
    for i in 0..config.num_particles {
        // arrange in cube arround 0,0
        let transform = get_position_in_grid(&config, i);
        let initial_color_material = materials.add(ColorMaterial::from(Color::from(PURPLE)));
        commands.spawn((
            MaterialMesh2dBundle {
                mesh: meshes
                    .add(new_circle(config.smoothing_radius * CIRCLE_RATIO))
                    .into(),
                material: initial_color_material,
                transform,
                ..default()
            },
            PredictedPosition(transform.translation.clone()),
            Velocity(Vec2::ZERO),
            Density {
                far: MASS,
                near: MASS,
            },
            Particle,
        ));
    }

    // a grid with squares for each pixel of the window
    /*
    let width = window.width();
    let height = window.height();
    println!("width: {}, height: {}", width, height);
    let size: f32 = 10.;
    let num_x = (width / size + 1.) as i32;
    let num_y = (height / size + 1.) as i32;

    // fill x and y with quads with size size
    for x in 0..num_x {
        for y in 0..num_y {
            let x = x as f32 * size - (width / 2.) + size / 2.;
            let y = y as f32 * size - (height / 2.) + size / 2.;
            commands.spawn((
                MaterialMesh2dBundle {
                    mesh: meshes
                        .add(shape::Quad::new(Vec2::new(size / 2., size / 2.)).into())
                        .into(),
                    material: materials.add(ColorMaterial::from(Color::WHITE)),
                    transform: Transform::from_translation(Vec3::new(x, y, -0.2)),
                    ..default()
                },
                Density {
                    far: MASS,
                    near: MASS,
                },
            ));
        }
    }
         */
}

/*
fn sync_meshes_system(
    config: Res<Config>,
    mut last_smoothing_radius: Local<f32>,
    mut first_run: Local<bool>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut particles_query: Query<&mut Mesh2dHandle, With<Particle>>,
) {
    if !*first_run {
        *first_run = true;
        *last_smoothing_radius = config.smoothing_radius;
        return;
    }

        if config.smoothing_radius == *last_smoothing_radius {
            return;
        }
        *last_smoothing_radius = config.smoothing_radius;

        for mut mesh in &mut particles_query {
            let old_id = mesh.0.clone();
            *mesh = meshes
            .add(new_circle(config.smoothing_radius * CIRCLE_RATIO))
            .into();
        meshes.remove(old_id);
    }
}

fn color_system(
    config: Res<Config>,
    mut particles_query: Query<
        (&Velocity, &Transform, &Density, &mut Handle<ColorMaterial>),
        With<Particle>,
    >,
    particles_query_inner: Query<&Transform, With<Particle>>,
    mut quads_query: Query<(&Density, &mut Handle<ColorMaterial>), Without<Particle>>,
    gradient_resource: Res<GradientResource>,
    color_scheme_categorical_resource: Res<ColorSchemeCategoricalResource>,
    spatial_hash: Res<SpatialHash>,
    mut gizmos: Gizmos,
) {
    let first_entity_id = spatial_hash.first_entity_id;

    if config.mark_sample_particle_neighbors && first_entity_id != Entity::from_raw(0) {
        let (_, transform, _, _) = particles_query.get(first_entity_id).unwrap();

        let cell = get_cell_2d(transform.translation.truncate(), config.smoothing_radius);
        let hash = hash_cell_2d(cell);
        let key = key_from_hash(hash, spatial_hash.indices.len() as u32);

        //println!("cell: {:?}  hash: {}  key: {}", cell, hash, key);

        let cell_color = color_scheme_categorical_resource
            .get_color_wrapped(&(key as usize))
            .clone();
        let cell_color = Color::rgba(cell_color.r(), cell_color.g(), cell_color.b(), 0.6);

        // draw circle with smoothing_radius around particle0
        gizmos.circle_2d(
            transform.translation.truncate(),
            config.smoothing_radius,
            Color::rgba(1., 1., 1., 0.3),
        );

        process_neighbors(
            &transform.translation,
            &spatial_hash,
            &config,
            |neighbor_entity_id| {
                let position2 = particles_query_inner.get(neighbor_entity_id).unwrap();

                let offset = position2.translation - transform.translation;
                let sqrt_dst = offset.length_squared();

                // skip if too far
                if sqrt_dst > config.smoothing_radius.powf(2.0) {
                    return;
                }

                // draw line to each neighbor
                gizmos.line_2d(
                    position2.translation.truncate(),
                    transform.translation.truncate(),
                    cell_color,
                );
            },
            Some(first_entity_id),
            false,
        );
    }

    if config.is_paused {
        return;
    }

    if config.particle_color_mode != ParticleColorMode::Blue {
        particles_query
            .par_iter_mut()
            .for_each(|(velocity, transform, density, mut material)| {
                if config.particle_color_mode == ParticleColorMode::Velocity {
                    let speed_normalized = velocity.0.length() / config.max_velocity_for_color;
                    //println!("speed_normalized {}", speed_normalized);
                    *material = gradient_resource.get_gradient_color_material(&speed_normalized);
                } else if config.particle_color_mode == ParticleColorMode::Density {
                    let density_normalized = density.far / config.max_density_for_color;
                    //println!("density_normalized {}", density_normalized);
                    *material = gradient_resource.get_gradient_color_material(&density_normalized);
                } else if config.particle_color_mode == ParticleColorMode::CellKey
                    && spatial_hash.indices.len() > 0
                {
                    let cell =
                        get_cell_2d(transform.translation.truncate(), config.smoothing_radius);
                    let hash = hash_cell_2d(cell);
                    let key = key_from_hash(hash, spatial_hash.indices.len() as u32);
                    let wrapped_color_index =
                        color_scheme_categorical_resource.get_wrapped_index(&(key as usize));
                    *material = color_scheme_categorical_resource
                        .get_color_material_wrapped(&wrapped_color_index);
                }
            });

        quads_query
            .par_iter_mut()
            .for_each(|(density, mut material)| {
                let density_normalized = density.far / config.max_density_for_color;
                *material = gradient_resource.get_gradient_color_material(&density_normalized);
            });
    }
}
 */
