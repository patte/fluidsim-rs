use std::sync::atomic::{AtomicBool, Ordering};

use bevy::{prelude::*, sprite::MaterialMesh2dBundle};
use bevy_internal::{
    //diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    input::common_conditions::input_toggle_active,
    window::PresentMode,
};

use bevy::window::Window;

use bevy_inspector_egui::{
    bevy_egui::EguiPlugin, prelude::ReflectInspectorOptions, quick::WorldInspectorPlugin,
    DefaultInspectorConfigPlugin, InspectorOptions,
};

use chrono::prelude::Utc;
use rand::{thread_rng, Rng};

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
}

fn default_particle_color_mode() -> ParticleColorMode {
    ParticleColorMode::Velocity
}

#[derive(Resource, Default)]
pub struct Measurements {
    delta_t: f32,
    tps: f32,
}

#[derive(Component)]
struct Particle;

#[derive(Component, Clone)]
struct Velocity(Vec2);

fn default_max_velocity_for_color() -> f32 {
    1000.0
}

#[derive(Component, Clone)]
struct Density {
    far: f32,
    near: f32,
}

fn default_target_density() -> f32 {
    2000.0
}

fn default_max_density_for_color() -> f32 {
    default_target_density()
}

#[derive(Component, Clone)]
struct PredictedPosition(Vec3);

#[derive(Default, Clone, Debug)]
struct SpatialIndex {
    index: u32,
    key: u32,
    hash: u32,
}

#[derive(Resource)]
struct SpatialHash {
    indices: Vec<SpatialIndex>,
    offsets: Vec<u32>,
    particles: Vec<(Vec3, PredictedPosition, Velocity, Density)>,
}

#[derive(
    Resource, Reflect, InspectorOptions, serde::Serialize, serde::Deserialize, Debug, Clone, Copy,
)]
#[reflect(Resource, InspectorOptions)]
pub struct Config {
    #[inspector(min = 0, max = 5000, speed = 1.)]
    num_particles: usize,
    gravity: Vec2,
    #[inspector(min = 0.0, max = 1.0, speed = 0.001)]
    damping: f32,
    #[inspector(min = 0.0, speed = 10.)]
    target_density: f32,
    #[inspector(min = 0.0, speed = 10.)]
    pressure_multiplier: f32,
    #[inspector(min = 0.0, speed = 10.)]
    #[serde(default)]
    near_pressure_multiplier: f32,
    #[inspector(min = 0.0, max = 1000.0, speed = 0.1)]
    smoothing_radius: f32,
    #[inspector(min = 0.0, max = 10000.0, speed = 1.)]
    #[serde(default = "default_max_velocity_for_color")]
    max_velocity_for_color: f32,
    #[inspector(min = 0.0000000000001, speed = 0.001)]
    #[serde(default = "default_max_density_for_color")]
    max_density_for_color: f32,
    #[serde(default = "default_particle_color_mode")]
    particle_color_mode: ParticleColorMode,
    #[serde(default)]
    mark_sample_particle_neighbors: bool,
    is_paused: bool,
    start_time: i64,
    auto_save: bool,
}

const RADIUS: f32 = 4.;
const MASS: f32 = 1.;
const TIME_STEP: f64 = 1. / 400.;

impl Default for Config {
    fn default() -> Self {
        Self {
            gravity: Vec2::new(0., 0.),
            damping: 0.05,
            target_density: default_target_density(),
            pressure_multiplier: 900.,
            near_pressure_multiplier: 2.,
            smoothing_radius: RADIUS * 12.,
            max_velocity_for_color: default_max_velocity_for_color(),
            max_density_for_color: default_max_density_for_color(),
            num_particles: 300,
            particle_color_mode: default_particle_color_mode(),
            mark_sample_particle_neighbors: false,
            is_paused: false,
            start_time: Utc::now().timestamp(),
            auto_save: false,
        }
    }
}

fn main() {
    // load most recent config from file
    let config = load_most_recent_config_from_file();

    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "🌊".into(),
                    present_mode: PresentMode::AutoNoVsync,
                    resolution: [1000., 1000.].into(),
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
        //.add_plugins(ResourceInspectorPlugin::<Config>::default())
        .insert_resource(config)
        .register_type::<Config>()
        .insert_resource(GradientResource::new())
        .insert_resource(ColorSchemeCategoricalResource::new())
        .insert_resource(SpatialHash {
            indices: Vec::<SpatialIndex>::new(),
            offsets: Vec::new(),
            particles: Vec::new(),
        })
        .insert_resource(Measurements::default())
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (inspector_ui, keyboard_animation_control, color_system),
        )
        .add_systems(
            FixedUpdate,
            (
                measurements_system,
                gravity_system,
                update_spatial_hash_system,
                calculate_density_system,
                pressure_force_system,
                move_system,
                bounce_system,
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
    windows: Query<&Window>,
    config: Res<Config>,
) {
    let window = windows.single();
    commands.spawn(Camera2dBundle::default());

    gradient_resource.precompute_materials(&mut materials);
    color_scheme_categorical_resource.precompute_materials(&mut materials);

    // spawn particles
    for _ in 0..config.num_particles {
        let transform = get_random_transform(&window);
        commands.spawn((
            MaterialMesh2dBundle {
                mesh: meshes.add(new_circle().into()).into(),
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
) {
    if config.is_paused {
        return;
    }

    measurements.delta_t = time.delta_seconds();
    measurements.tps = 1. / measurements.delta_t;
}

fn gravity_system(
    time: Res<Time>,
    config: Res<Config>,
    mut particles_query: Query<(&Transform, &mut PredictedPosition, &mut Velocity), With<Particle>>,
) {
    if config.is_paused {
        return;
    }

    let delta_t = time.delta_seconds();

    particles_query
        .par_iter_mut()
        .for_each(|(transform, mut predicted_position, mut velocity)| {
            // print on direction change
            //if velocity.0.y > 0. && velocity.0.y + config.gravity.y * delta_t < 0.
            //    || velocity.0.y < 0. && velocity.0.y + config.gravity.y * delta_t > 0.
            //{
            //    println!("t translation: {:?}", transform.translation);
            //}
            velocity.0 += config.gravity * delta_t;
            // fixed time step
            predicted_position.0 = transform.translation; // + velocity.0.extend(0.) * (1. / 1600.);
        });
}

fn update_spatial_hash_system(
    mut spatial_hash: ResMut<SpatialHash>,
    mut particles_query: Query<
        (&Transform, &PredictedPosition, &Density, &Velocity),
        With<Particle>,
    >,
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
                index: u32::MAX,
                key: u32::MAX,
                hash: u32::MAX,
            },
        );
        spatial_hash.offsets.resize(num_particles, u32::MAX);
        println!("spatial_hash.indices.len(): {}", spatial_hash.indices.len());
    }

    // reset offsets
    spatial_hash
        .offsets
        .iter_mut()
        .for_each(|offset| *offset = u32::MAX);

    spatial_hash.particles.clear();

    // indices
    let mut new_indices: Vec<SpatialIndex> = Vec::new();

    for (i, (transform, predicted_position, density, velocity)) in
        particles_query.iter_mut().enumerate()
    {
        let cell = get_cell_2d(predicted_position.0.truncate(), config.smoothing_radius);
        let hash = hash_cell_2d(cell);
        let key = key_from_hash(hash, spatial_hash.indices.len() as u32);
        //println!("cell: {:?}  hash: {}  key: {}", cell, hash, key);
        //spatial_hash.indices[i] = SpatialIndex {
        new_indices.push(SpatialIndex {
            index: i as u32,
            key,
            hash,
        });
        spatial_hash.particles.push((
            transform.translation.clone(),
            predicted_position.clone(),
            velocity.clone(),
            density.clone(),
        ));
    }

    // offsets
    new_indices.sort_by(|a, b| a.key.partial_cmp(&b.key).unwrap());

    let mut last_key = u32::MAX;
    // set spatial_hash.offsets to the first index of each hash
    new_indices.iter().enumerate().for_each(|(i, index)| {
        if index.key != last_key {
            spatial_hash.offsets[index.key as usize] = i as u32;
            last_key = index.key;
        }
    });

    spatial_hash.indices = new_indices;

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
    particles_query2: Query<&PredictedPosition, With<Particle>>,
    config: Res<Config>,
    spatial_hash: Res<SpatialHash>,
) {
    if config.is_paused {
        return;
    }

    //let mut all_densities = Vec::new();
    //for (mut density, transform) in &mut particles_query {

    // shared mutable variable
    //let marked = AtomicBool::new(true);

    particles_query
        .par_iter_mut()
        .for_each(|(predicted_position, mut density)| {
            let mut density_sum = 0.;
            let mut density_near_sum = 0.;

            let mut density_sum2 = 0.;
            let mut density_near_sum2 = 0.;

            // use process_neighbors function
            process_neighbors(
                &predicted_position.0,
                &spatial_hash,
                &config,
                |_, neighbor_predicted_position, _, _| {
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
                false,
            );

            /*
            if !marked.load(Ordering::Relaxed) {
                marked.store(true, Ordering::Relaxed);

                for transform2 in &particles_query2 {
                    if transform.translation == transform2.translation {
                        continue;
                    }
                    let distance = (transform2.translation - transform.translation).length();

                    if distance.powf(2.0) > config.smoothing_radius.powf(2.0) {
                        continue;
                    }

                    density_sum2 += spiky_kernel_pow_2(&config.smoothing_radius, &distance);
                    density_near_sum2 += spiky_kernel_pow_3(&config.smoothing_radius, &distance);
                }

                if density_sum != density_sum2 || density_near_sum != density_near_sum2 {
                    println!(
                        "d_sum: {}  d_near_sum: {} density_sum: {} error%_sum: {}  error%_near_sum: {}",
                        density_sum - density_sum2,
                        density_near_sum - density_near_sum2,
                        density_sum,
                        100.* ((density_sum - density_sum2) / density_sum),
                        100.*((density_near_sum - density_near_sum2) / density_near_sum)
                    );
                }
            }
            */

            // TODO: check if others need scaling to, extract into config
            density.far = density_sum * 1000.;
            density.near = density_near_sum * 1000.;

            //all_densities.push(density.far);
        });

    // average density
    //let average_density = all_densities.iter().sum::<f32>() / all_densities.len() as f32;
    //println!(
    //    "average_density: {}  min_density: {}  max_density: {}",
    //    average_density,
    //    all_densities
    //        .iter()
    //        .min_by(|a, b| a.partial_cmp(b).unwrap())
    //        .unwrap(),
    //    all_densities
    //        .iter()
    //        .max_by(|a, b| a.partial_cmp(b).unwrap())
    //        .unwrap()
    //);
}

fn pressure_force_system(
    time: Res<Time>,
    mut particles_query: Query<(&PredictedPosition, &mut Velocity, &Density), With<Particle>>,
    config: Res<Config>,
    spatial_hash: Res<SpatialHash>,
) {
    if config.is_paused {
        return;
    }
    let delta_t = time.delta_seconds();

    let pressure_from_density = |density: &f32| -> f32 {
        return (density - config.target_density) * config.pressure_multiplier;
    };

    let near_pressure_from_density = |density: &f32| -> f32 {
        return -density * config.near_pressure_multiplier;
    };

    let mut rng = thread_rng();
    let random_direction =
        Vec3::new(rng.gen_range(-1. ..1.), rng.gen_range(-1. ..1.), 0.).normalize();

    particles_query
        .par_iter_mut()
        .for_each(|(predicted_position, mut velocity, density)| {
            let mut sum_pressure_force = Vec3::ZERO;

            process_neighbors(
                &predicted_position.0,
                &spatial_hash,
                &config,
                |_, predicted_position2, _, density2| {
                    let offset = predicted_position2.0 - predicted_position.0;
                    let sqrt_dst = offset.length_squared();

                    // skip if too far
                    if sqrt_dst > config.smoothing_radius.powf(2.0) {
                        return;
                    }

                    let distance = sqrt_dst.sqrt();
                    let direction = if distance > 0. {
                        offset / distance
                    } else {
                        random_direction
                    };

                    let shared_pressure = (pressure_from_density(&density.far)
                        + pressure_from_density(&density2.far))
                        * 0.5;
                    let shared_pressure_near = (near_pressure_from_density(&density.near)
                        + near_pressure_from_density(&density2.near))
                        * 0.5;

                    sum_pressure_force += -direction
                        * derivative_spiky_pow_2(&config.smoothing_radius, &distance)
                        * shared_pressure
                        / density2.far.max(1.);

                    sum_pressure_force += -direction
                        * derivative_spiky_pow_3(&config.smoothing_radius, &distance)
                        * shared_pressure_near
                        / density2.near.max(1.);
                },
                false,
            );

            let acceleration = sum_pressure_force / (density.far * density.near).max(1.);
            //println!("acceleration {:?}", acceleration);
            velocity.0.x += acceleration.x * delta_t;
            velocity.0.y += acceleration.y * delta_t;
        });
}

fn move_system(
    config: Res<Config>,
    time: Res<Time>,
    mut particles_query: Query<(&mut Transform, &Velocity), With<Particle>>,
) {
    if config.is_paused {
        return;
    }
    let delta_t = time.delta_seconds();

    particles_query
        .par_iter_mut()
        .for_each(|(mut transform, velocity)| {
            transform.translation += velocity.0.extend(0.) * delta_t;
            //println!("t translation: {:?}", transform.translation);
        });
}

fn bounce_system(
    config: Res<Config>,
    windows: Query<&Window>,
    mut particles_query: Query<(&mut Transform, &mut Velocity), With<Particle>>,
) {
    if config.is_paused {
        return;
    }
    const MARGIN: f32 = 0.;

    let window = windows.single();
    let width = window.width() - MARGIN;
    let height = window.height() - MARGIN;

    let half_size = Vec3::new(width / 2., height / 2., 0.0);

    particles_query
        .par_iter_mut()
        .for_each(|(mut transform, mut velocity)| {
            let edge_dst = half_size - (transform.translation.abs() + Vec3::splat(RADIUS));

            if edge_dst.x <= 0. {
                // switch direction
                if velocity.0.x.signum() == transform.translation.x.signum() {
                    velocity.0.x *= -1. * (1.0 - config.damping);
                }

                // if on the edge, apply force inwards
                // TODO: check this again
                /*
                if edge_dst.x > -1. {
                    velocity.0.x += -transform.translation.x.signum() * 2.;
                }
                */

                // move inside
                transform.translation.x += -transform.translation.x.signum() * edge_dst.x.abs();
            }
            if edge_dst.y <= 0. {
                // switch direction
                if velocity.0.y.signum() == transform.translation.y.signum() {
                    velocity.0.y = -velocity.0.y * (1.0 - config.damping);
                }

                // if on the edge, apply force inwards
                // TODO: check this again
                /*
                if edge_dst.y > -1. {
                    velocity.0.y += -transform.translation.y.signum() * 2.;
                }
                */

                // move inside
                transform.translation.y -= transform.translation.y.signum() * edge_dst.y.abs();
            }
        });
}

// color particles based on their velocity
// color quads based on their density
fn color_system(
    config: Res<Config>,
    mut particles_query: Query<
        (&Velocity, &Transform, &Density, &mut Handle<ColorMaterial>),
        With<Particle>,
    >,
    mut quads_query: Query<(&Density, &mut Handle<ColorMaterial>), Without<Particle>>,
    gradient_resource: Res<GradientResource>,
    color_scheme_categorical_resource: Res<ColorSchemeCategoricalResource>,
    spatial_hash: Res<SpatialHash>,
    mut gizmos: Gizmos,
) {
    if config.mark_sample_particle_neighbors && spatial_hash.particles.len() > 0 {
        let (particle0_position, _, _, _) = spatial_hash.particles.get(0).unwrap();
        let cell = get_cell_2d(particle0_position.truncate(), config.smoothing_radius);
        let hash = hash_cell_2d(cell);
        let key = key_from_hash(hash, spatial_hash.indices.len() as u32);

        let cell_color = color_scheme_categorical_resource
            .get_color_wrapped(&(key as usize))
            .clone();
        let cell_color = Color::rgba(cell_color.r(), cell_color.g(), cell_color.b(), 0.6);

        // draw circle with smoothing_radius around particle0
        gizmos.circle_2d(
            particle0_position.truncate(),
            config.smoothing_radius,
            Color::rgba(1., 1., 1., 0.3),
        );

        process_neighbors(
            &particle0_position,
            &spatial_hash,
            &config,
            |neighbor_position, _, _, _| {
                // draw line to each neighbor
                gizmos.line_2d(
                    neighbor_position.truncate(),
                    particle0_position.truncate(),
                    cell_color,
                );
            },
            false,
        );
    }
    if config.is_paused {
        return;
    }

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
                let cell = get_cell_2d(transform.translation.truncate(), config.smoothing_radius);
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

fn keyboard_animation_control(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    keyboard_input: Res<Input<KeyCode>>,
    mut config: ResMut<Config>,
    mut particles_query: Query<(&mut Velocity, &mut Transform), With<Particle>>,
    windows: Query<&Window>,
) {
    let window = windows.single();
    let mut key_pressed = false;
    /*
    let gravity_delta = 0.05;
    if keyboard_input.pressed(KeyCode::Up) {
        config.gravity.y += gravity_delta;
        key_pressed = true;
    }
    if keyboard_input.pressed(KeyCode::Down) {
        config.gravity.y -= gravity_delta;
        key_pressed = true;
    }
    if keyboard_input.pressed(KeyCode::Left) {
        config.gravity.x -= gravity_delta;
        key_pressed = true;
    }
    if keyboard_input.pressed(KeyCode::Right) {
        config.gravity.x += gravity_delta;
        key_pressed = true;
    }

    let damping_delta = 0.001;
    if keyboard_input.pressed(KeyCode::W) {
        config.damping += damping_delta;
        key_pressed = true;
    }
    if keyboard_input.pressed(KeyCode::S) {
        config.damping -= damping_delta;
        key_pressed = true;
    }
    if keyboard_input.pressed(KeyCode::X) {
        config.damping = DAMPING_DEFAULT;
        key_pressed = true;
    }

    let target_density_delta = 0.2;
    if keyboard_input.pressed(KeyCode::E) {
        config.target_density += target_density_delta;
        key_pressed = true;
    }
    if keyboard_input.pressed(KeyCode::D) {
        config.target_density -= target_density_delta;
        key_pressed = true;
    }
    if keyboard_input.pressed(KeyCode::C) {
        config.target_density = TARGET_DENSITY_DEFAULT;
        key_pressed = true;
    }

    let pressure_multiplier_delta = 5.;
    if keyboard_input.pressed(KeyCode::R) {
        config.pressure_multiplier += pressure_multiplier_delta;
        key_pressed = true;
    }
    if keyboard_input.pressed(KeyCode::F) {
        config.pressure_multiplier -= pressure_multiplier_delta;
        key_pressed = true;
    }
    if keyboard_input.pressed(KeyCode::V) {
        config.pressure_multiplier = PRESSURE_MULTIPLIER_DEFAULT;
        key_pressed = true;
    }

    let smoothing_radius_delta = 0.5;
    if keyboard_input.pressed(KeyCode::T) {
        config.smoothing_radius += smoothing_radius_delta;
        key_pressed = true;
    }
    if keyboard_input.pressed(KeyCode::G) {
        config.smoothing_radius -= smoothing_radius_delta;
        key_pressed = true;
    }
    if keyboard_input.pressed(KeyCode::B) {
        config.smoothing_radius = SMOOTHING_RADIUS_DEFAULT;
        key_pressed = true;
    }
     */

    // reset position
    if keyboard_input.just_pressed(KeyCode::Space) {
        for (mut velocity, mut transform) in &mut particles_query {
            velocity.0 = Vec2::ZERO;
            transform.translation = get_random_transform(window).translation;
        }
        key_pressed = true;
    }

    // pause simulation
    if keyboard_input.just_pressed(KeyCode::P) {
        config.is_paused = !config.is_paused;
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
                    mesh: meshes.add(new_circle().into()).into(),
                    material: materials.add(ColorMaterial::from(Color::PURPLE)),
                    transform: get_random_transform(window),
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

fn process_neighbors<F>(
    me_position: &Vec3,
    spatial_hash: &SpatialHash,
    config: &Config,
    mut process: F,
    mark: bool,
) where
    F: FnMut(&Vec3, &PredictedPosition, &Velocity, &Density),
{
    let original_cell = get_cell_2d(me_position.truncate(), config.smoothing_radius);
    let original_hash = hash_cell_2d(original_cell);
    let original_key = key_from_hash(original_hash, spatial_hash.indices.len() as u32);

    if mark {
        println!(
            "original_cell: {:?}  hash: {}  key: {}",
            original_cell, original_hash, original_key
        );
    }

    for offset in OFFSETS_2D.iter() {
        let cell = original_cell + *offset;
        let hash = hash_cell_2d(cell);
        let key = key_from_hash(hash, spatial_hash.indices.len() as u32);

        if mark {
            println!(
                "  offset: {} cell: {:?}  hash: {}  key: {}",
                offset, cell, hash, key
            );
        }

        if let Some(&start_index) = spatial_hash.offsets.get(key as usize) {
            let mut self_processed = false;
            for i in start_index as usize..spatial_hash.indices.len() - 1 {
                if mark {
                    println!("    start_index: {} i: {}", start_index, i);
                }
                let index_data = spatial_hash.indices.get(i as usize).unwrap();

                if index_data.key != key {
                    break;
                }
                if index_data.hash != hash {
                    continue;
                }
                let (
                    neighbor_position,
                    neighbor_predicted_position,
                    neighbor_velocity,
                    neighbor_density,
                ) = spatial_hash
                    .particles
                    .get(index_data.index as usize)
                    .unwrap();

                if me_position.x == neighbor_position.x && me_position.y == neighbor_position.y {
                    if self_processed {
                        continue;
                    }
                    self_processed = true;
                    continue;
                }

                process(
                    neighbor_position,
                    neighbor_predicted_position,
                    neighbor_velocity,
                    neighbor_density,
                );
            }
        }
    }
}
