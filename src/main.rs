use bevy::{prelude::*, sprite::MaterialMesh2dBundle};
use bevy_internal::{
    //diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    input::common_conditions::input_toggle_active,
    sprite::Mesh2dHandle,
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

use bevy_hanabi::Gradient;

mod ui;
use ui::*;

mod file_io;
use file_io::*;

mod utils;
use utils::*;

mod colors;

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
    40.
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
}

const MASS: f32 = 1.;
const TIME_STEP: f64 = 1. / 180.;
pub const SCALE_FACTOR: f32 = 0.02;
const CIRCLE_RATIO: f32 = 0.10;

impl Default for Config {
    fn default() -> Self {
        Self {
            gravity: Vec2::new(0., -9.81),
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
            interaction_input_strength: 60.,
            interaction_input_radius: 2.,
            time_scale: 1.,
            prediction_time_scale: 0.5,
            is_paused: false,
            pause_after_next_frame: false,
            start_time: Utc::now().timestamp(),
            auto_save: false,
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
                    resolution: [1800., 1000.].into(),
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
        .insert_resource(ClearColor(Color::rgb(0.0, 0.0, 0.0)))
        .insert_resource(config)
        .register_type::<Config>()
        .insert_resource(GradientResource::new())
        .insert_resource(ColorSchemeCategoricalResource::new())
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
                mouse_interaction_system,
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
                sync_meshes_system,
                bounce_system,
                color_system,
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
    mut gradient_resource: ResMut<GradientResource>,
    mut color_scheme_categorical_resource: ResMut<ColorSchemeCategoricalResource>,
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

    gradient_resource.precompute_materials(&mut materials);
    color_scheme_categorical_resource.precompute_materials(&mut materials);

    // spawn particles
    for i in 0..config.num_particles {
        // arrange in cube arround 0,0
        let transform = get_position_in_grid(&config, i);
        commands.spawn((
            MaterialMesh2dBundle {
                mesh: meshes
                    .add(new_circle(config.smoothing_radius * CIRCLE_RATIO).into())
                    .into(),
                material: materials.add(ColorMaterial::from(Color::PURPLE)),
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

fn measurements_system(
    time: Res<Time>,
    mut measurements: ResMut<Measurements>,
    config: Res<Config>,
    particles_query: Query<(&Transform, &PredictedPosition, &Velocity, &Density), With<Particle>>,
    spatial_hash: Res<SpatialHash>,
) {
    if config.mark_sample_particle_neighbors && spatial_hash.first_entity_id != Entity::from_raw(0)
    {
        let (p0_position, p0_predicted_position, p0_velocity, p0_density) =
            particles_query.get(spatial_hash.first_entity_id).unwrap();
        measurements.p0_position = p0_position.translation.clone();
        measurements.p0_predicted_position = p0_predicted_position.0.clone();
        measurements.p0_velocity = p0_velocity.0.clone();
        measurements.p0_density = p0_density.clone();
        if p0_density.far > measurements.p0_max_density_far {
            measurements.p0_max_density_far = p0_density.far;
        }
    }

    if config.is_paused {
        return;
    }

    measurements.delta_t = time.delta_seconds() * config.time_scale;
    measurements.tps = 1. / measurements.delta_t;
}

fn gravity_system(
    time: Res<Time>,
    config: Res<Config>,
    mut particles_query: Query<(&Transform, &mut PredictedPosition, &mut Velocity), With<Particle>>,
    interaction_inputs: Res<InteractionInputs>,
) {
    if config.is_paused {
        return;
    }

    let delta_t = time.delta_seconds() * config.time_scale;

    particles_query
        .par_iter_mut()
        .for_each(|(transform, mut predicted_position, mut velocity)| {
            let mut acceleration = config.gravity;

            if interaction_inputs.strength != 0. {
                let input_point_offset =
                    interaction_inputs.point.unwrap() - transform.translation.xy();
                let sqr_dst = input_point_offset.length_squared();
                if sqr_dst < config.interaction_input_radius.powf(2.) {
                    let dst = sqr_dst.sqrt();
                    let edge_t = dst / config.interaction_input_radius;
                    let centre_t = 1. - edge_t;
                    let dir_to_centre = input_point_offset / dst;

                    let gravity_weight = 1. - (centre_t * (interaction_inputs.strength / 10.));

                    acceleration = acceleration * gravity_weight
                        + dir_to_centre * centre_t * interaction_inputs.strength;
                    acceleration -= velocity.0 * centre_t;
                }
            }

            velocity.0 += acceleration * delta_t;
            predicted_position.0 = transform.translation
                + velocity.0.extend(0.) * (delta_t as f32 * config.prediction_time_scale);
        });
}

fn update_spatial_hash_system(
    mut spatial_hash: ResMut<SpatialHash>,
    mut particles_query: Query<(Entity, &PredictedPosition), With<Particle>>,
    config: Res<Config>,
) {
    if config.is_paused {
        return;
    }

    let num_particles = particles_query.iter_mut().len();

    // resize
    if num_particles > spatial_hash.indices.len() {
        spatial_hash.indices.resize(
            num_particles,
            SpatialIndex {
                key: u32::MAX,
                hash: u32::MAX,
                entity_id: Entity::from_raw(0),
            },
        );
        spatial_hash.offsets.resize(num_particles, usize::MAX);
        println!("spatial_hash.indices.len(): {}", spatial_hash.indices.len());
    }

    // new indices
    let mut new_indices: Vec<SpatialIndex> = Vec::new();

    // remember first entity id
    let mut first_entity_id = Entity::from_raw(0);

    for (entity_id, predicted_position) in particles_query.iter_mut() {
        let cell = get_cell_2d(predicted_position.0.truncate(), config.smoothing_radius);
        let hash = hash_cell_2d(cell);
        let key = key_from_hash(hash, spatial_hash.indices.len() as u32);
        new_indices.push(SpatialIndex {
            key,
            hash,
            entity_id,
        });
        if first_entity_id == Entity::from_raw(0) {
            first_entity_id = entity_id;
        }
    }

    // sort by key
    new_indices.sort_by(|a, b| a.key.partial_cmp(&b.key).unwrap());

    // reset offsets
    spatial_hash
        .offsets
        .iter_mut()
        .for_each(|offset| *offset = usize::MAX);

    // set spatial_hash.offsets to the first index of each hash
    let mut last_key = u32::MAX;
    new_indices.iter().enumerate().for_each(|(i, index)| {
        if index.key != last_key {
            spatial_hash.offsets[index.key as usize] = i;
            last_key = index.key;
        }
    });

    spatial_hash.indices = new_indices;

    spatial_hash.first_entity_id = first_entity_id;

    // print first 20 entries
    //println!(
    //    "spatial_hash.indices: {:?}",
    //    &spatial_hash.indices[0..80.min(spatial_hash.indices.len())]
    //);
    //println!(
    //    "spatial_hash.offsets: {:?}",
    //    &spatial_hash.offsets[0..20.min(spatial_hash.offsets.len())]
    //);
}

fn calculate_density_system(
    mut particles_query: Query<(&PredictedPosition, &mut Density), With<Density>>,
    particles_query_inner: Query<&PredictedPosition, With<Particle>>,
    config: Res<Config>,
    spatial_hash: Res<SpatialHash>,
) {
    if config.is_paused {
        return;
    }

    particles_query
        .par_iter_mut()
        .for_each(|(predicted_position, mut density)| {
            let mut density_sum = 0.;
            let mut density_near_sum = 0.;

            process_neighbors(
                &predicted_position.0,
                &spatial_hash,
                &config,
                |neighbor_entity_id| {
                    let neighbor_predicted_position =
                        particles_query_inner.get(neighbor_entity_id).unwrap();

                    let sqrt_dst =
                        (neighbor_predicted_position.0 - predicted_position.0).length_squared();

                    // skip if too far
                    if sqrt_dst > config.smoothing_radius.powf(2.0) {
                        return;
                    }

                    let distance = sqrt_dst.sqrt();

                    density_sum += spiky_kernel_pow_2(&config.smoothing_radius, &distance);
                    density_near_sum += spiky_kernel_pow_3(&config.smoothing_radius, &distance);
                },
                None, // include self
                false,
            );

            density.far = density_sum;
            density.near = density_near_sum;
        });
}

fn pressure_force_system(
    time: Res<Time>,
    mut particles_query: Query<
        (Entity, &PredictedPosition, &mut Velocity, &Density),
        With<Particle>,
    >,
    particles_query_inner: Query<(&PredictedPosition, &Density), With<Particle>>,
    config: Res<Config>,
    spatial_hash: Res<SpatialHash>,
) {
    if config.is_paused {
        return;
    }
    let delta_t = time.delta_seconds() * config.time_scale;

    let pressure_from_density = |density: f32| -> f32 {
        return (density - config.target_density) * config.pressure_multiplier;
    };

    let near_pressure_from_density = |density: &f32| -> f32 {
        return density * config.near_pressure_multiplier;
    };

    //let mut rng = thread_rng();
    //let random_direction = Vec2::new(rng.gen_range(-1. ..1.), rng.gen_range(-1. ..1.)).normalize();
    let random_direction = Vec2::new(0., 1.);

    particles_query.par_iter_mut().for_each(
        |(entity_id, predicted_position, mut velocity, density)| {
            let mut sum_pressure_force = Vec2::ZERO;
            let pressure = pressure_from_density(density.far);
            let near_pressure = near_pressure_from_density(&density.near);

            process_neighbors(
                &predicted_position.0,
                &spatial_hash,
                &config,
                |neighbor_entity_id| {
                    let (predicted_position2, density2) =
                        particles_query_inner.get(neighbor_entity_id).unwrap();

                    let offset = predicted_position2.0 - predicted_position.0;
                    let sqrt_dst = offset.length_squared();

                    // skip if too far
                    if sqrt_dst > config.smoothing_radius.powf(2.0) {
                        return;
                    }

                    let distance = sqrt_dst.sqrt();
                    let direction = if distance > 0. {
                        offset.xy() / distance
                    } else {
                        random_direction
                    };

                    let shared_pressure = (pressure + pressure_from_density(density2.far)) * 0.5;
                    let shared_pressure_near =
                        (near_pressure + near_pressure_from_density(&density2.near)) * 0.5;

                    sum_pressure_force += direction
                        * derivative_spiky_pow_2(&config.smoothing_radius, &distance)
                        * shared_pressure
                        / density2.far;

                    sum_pressure_force += direction
                        * derivative_spiky_pow_3(&config.smoothing_radius, &distance)
                        * shared_pressure_near
                        / density2.near;
                },
                Some(entity_id), // exclude self
                false,
            );

            let acceleration = sum_pressure_force / density.far;

            velocity.0 += acceleration * delta_t;
        },
    );
}

fn move_system(
    config: Res<Config>,
    time: Res<Time>,
    mut particles_query: Query<(&mut Transform, &Velocity), With<Particle>>,
) {
    if config.is_paused {
        return;
    }
    let delta_t = time.delta_seconds() * config.time_scale;

    particles_query
        .par_iter_mut()
        .for_each(|(mut transform, velocity)| {
            transform.translation += velocity.0.extend(0.) * (delta_t);
        });
}

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
            .add(new_circle(config.smoothing_radius * CIRCLE_RATIO).into())
            .into();
        meshes.remove(old_id);
    }
}

fn bounce_system(
    mut config: ResMut<Config>,
    mut particles_query: Query<(&mut Transform, &mut Velocity), With<Particle>>,
) {
    if config.is_paused {
        return;
    }

    let width = config.bounding_box.width;
    let height = config.bounding_box.height;

    let half_size = Vec3::new(width / 2., height / 2., 0.0);

    particles_query
        .par_iter_mut()
        .for_each(|(mut transform, mut velocity)| {
            let edge_dst = half_size - transform.translation.abs();

            if edge_dst.x <= 0. {
                // switch direction
                if velocity.0.x.signum() == transform.translation.x.signum() {
                    velocity.0.x *= -1. * (1.0 - config.damping);
                }

                // move inside
                transform.translation.x += -transform.translation.x.signum() * edge_dst.x.abs();
            }
            if edge_dst.y <= 0. {
                // switch direction
                if velocity.0.y.signum() == transform.translation.y.signum() {
                    velocity.0.y = -velocity.0.y * (1.0 - config.damping);
                }

                // move inside
                transform.translation.y -= transform.translation.y.signum() * edge_dst.y.abs();
            }
        });

    if config.pause_after_next_frame {
        config.is_paused = true;
        config.pause_after_next_frame = false;
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

fn keyboard_interaction_system(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    keyboard_input: Res<Input<KeyCode>>,
    mut config: ResMut<Config>,
    mut particles_query: Query<(&mut Velocity, &mut Transform), With<Particle>>,
    mut measurements: ResMut<Measurements>,
) {
    let mut key_pressed = false;

    // reset position
    if keyboard_input.just_pressed(KeyCode::Space) {
        for (i, (mut velocity, mut transform)) in &mut particles_query.iter_mut().enumerate() {
            velocity.0 = Vec2::ZERO;
            transform.translation = get_position_in_grid(&config, i).translation;
        }
        measurements.p0_max_density_far = 0.;
        key_pressed = true;
    }

    // pause simulation
    if keyboard_input.just_pressed(KeyCode::P) {
        config.is_paused = !config.is_paused;
        key_pressed = true;
    }

    // pause after next frame
    if keyboard_input.just_pressed(KeyCode::Right) {
        config.is_paused = false;
        config.pause_after_next_frame = true;
        key_pressed = true;
    }

    // reset config
    if keyboard_input.just_pressed(KeyCode::I) {
        *config = Config::default();

        key_pressed = true;
    }

    // save config
    if keyboard_input.just_pressed(KeyCode::Z) {
        save_config_to_file(config.clone());
        println!("config saved!");
        key_pressed = true;
    }

    //load config
    if keyboard_input.just_pressed(KeyCode::U) {
        *config = load_most_recent_config_from_file();
        println!("config loaded!");
        key_pressed = true;
    }

    // toggle auto save
    if keyboard_input.just_pressed(KeyCode::K) {
        config.auto_save = !config.auto_save;
        println!("auto_save: {}", config.auto_save);
        key_pressed = true;
    }

    // pop new particle at random position
    if keyboard_input.just_pressed(KeyCode::N) || keyboard_input.just_pressed(KeyCode::M) {
        let spawn_num_particles = if keyboard_input.just_pressed(KeyCode::N) {
            1
        } else {
            10
        };
        for _ in 0..spawn_num_particles {
            commands.spawn((
                MaterialMesh2dBundle {
                    mesh: meshes
                        .add(new_circle(config.smoothing_radius * CIRCLE_RATIO).into())
                        .into(),
                    material: materials.add(ColorMaterial::from(Color::PURPLE)),
                    transform: get_random_transform(&config),
                    ..default()
                },
                PredictedPosition(Vec3::ZERO),
                Velocity(Vec2::new(0., 0.)),
                Density {
                    far: MASS,
                    near: MASS,
                },
                Particle,
            ));
        }
        config.num_particles += spawn_num_particles;
        key_pressed = true;
    }

    // print
    if key_pressed {
        println!(
            "gravity: [{} {}]  edge-damping: {}  target-density: {}  pressure-mult: {}  smoothing-radius: {}",
            config.gravity.x, config.gravity.y, 1. - config.damping, config.target_density, config.pressure_multiplier, config.smoothing_radius
        );
        if config.auto_save {
            save_config_to_file(config.clone());
        }
    }
}

fn mouse_interaction_system(
    q_windows: Query<&Window, With<bevy_internal::window::PrimaryWindow>>,
    buttons: Res<Input<MouseButton>>,
    config: Res<Config>,
    mut interaction_inputs: ResMut<InteractionInputs>,
) {
    let window = q_windows.single();
    if let Some(position) = window.cursor_position() {
        let x = (position.x - (window.width() / 2.)) * SCALE_FACTOR;
        let y = -(position.y - (window.height() / 2.)) * SCALE_FACTOR;
        let interaction_pos = Vec2::new(x, y);

        if buttons.pressed(MouseButton::Left) {
            interaction_inputs.point = Some(interaction_pos);
            interaction_inputs.strength = config.interaction_input_strength;
        } else if buttons.pressed(MouseButton::Right) {
            interaction_inputs.point = Some(interaction_pos);
            interaction_inputs.strength = -config.interaction_input_strength;
        } else {
            interaction_inputs.point = None;
            interaction_inputs.strength = 0.;
        }
    }
}

fn process_neighbors<F>(
    me_position: &Vec3,
    spatial_hash: &SpatialHash,
    config: &Config,
    mut process: F,
    skip_entity_id: Option<Entity>,
    log: bool,
) where
    F: FnMut(Entity),
{
    let original_cell = get_cell_2d(me_position.truncate(), config.smoothing_radius);
    let original_hash = hash_cell_2d(original_cell);
    let original_key = key_from_hash(original_hash, spatial_hash.indices.len() as u32);

    if log {
        println!(
            "original_cell: {:?}  hash: {}  key: {}",
            original_cell, original_hash, original_key
        );
    }

    for offset in OFFSETS_2D.iter() {
        let cell = original_cell + *offset;
        let hash = hash_cell_2d(cell);
        let key = key_from_hash(hash, spatial_hash.indices.len() as u32);

        if log && *offset == Vec2::new(0., 0.) {
            println!(
                "  offset: {} cell: {:?}  hash: {}  key: {}",
                offset, cell, hash, key
            );
        }

        if let Some(&start_index) = spatial_hash.offsets.get(key as usize) {
            // a great number of time has been spent here on a off by one error spatial_hash.indices.len()-1
            for i in start_index as usize..spatial_hash.indices.len() {
                let index_data = spatial_hash.indices.get(i as usize).unwrap();
                if log {
                    println!(
                        "    key {} start_index: {} i: {} index_data.index {:?}",
                        key, start_index, i, index_data.entity_id
                    );
                }

                if index_data.key != key {
                    break;
                }
                if index_data.hash != hash {
                    continue;
                }

                if skip_entity_id.is_some() && skip_entity_id.unwrap() == index_data.entity_id {
                    continue;
                }

                process(index_data.entity_id);
            }
        }
    }
}
