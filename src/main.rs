use std::{
    fs::{self, File},
    io::{Read, Write},
};

use bevy::{prelude::*, sprite::MaterialMesh2dBundle};
use bevy_internal::{
    //diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    input::common_conditions::input_toggle_active,
    window::PresentMode,
};

use bevy::window::PrimaryWindow;
use bevy::window::Window;

use bevy_inspector_egui::{
    bevy_egui::{EguiContext, EguiPlugin},
    egui::{self, FontId, RichText},
    prelude::ReflectInspectorOptions,
    quick::WorldInspectorPlugin,
    DefaultInspectorConfigPlugin, InspectorOptions,
};

use bevy_hanabi::Gradient;
use rand::{thread_rng, Rng};

mod math;
use math::*;

use serde_json;

use chrono::prelude::{DateTime, Utc};

#[derive(Component)]
struct Particle;

#[derive(Component)]
struct Velocity(Vec2);

#[derive(Component)]
struct Density {
    far: f32,
    near: f32,
}

#[derive(
    Resource, Reflect, InspectorOptions, serde::Serialize, serde::Deserialize, Debug, Clone, Copy,
)]
#[reflect(Resource, InspectorOptions)]
struct Config {
    #[inspector(min = 0, max = 5000, speed = 1.)]
    num_particles: usize,
    gravity: Vec2,
    #[inspector(min = 0.0, max = 1.0, speed = 0.001)]
    damping: f32,
    #[inspector(speed = 100.)]
    target_density: f32,
    #[inspector(speed = 100.)]
    pressure_multiplier: f32,
    #[inspector(min = 0.0, max = 1000.0, speed = 0.1)]
    smoothing_radius: f32,
    is_paused: bool,
    start_time: i64,
    auto_save: bool,
}

const RADIUS: f32 = 4.;
const MASS: f32 = 1.;

impl Default for Config {
    fn default() -> Self {
        Self {
            gravity: Vec2::new(0., 0.),
            damping: 0.05,
            target_density: 2000.,
            pressure_multiplier: 900.,
            smoothing_radius: RADIUS * 12.,
            num_particles: 300,
            is_paused: false,
            start_time: Utc::now().timestamp(),
            auto_save: false,
        }
    }
}

#[derive(Resource)]
struct GradientResource {
    gradient: Gradient<Vec4>,
    precomputed_materials: Vec<Handle<ColorMaterial>>,
}

impl GradientResource {
    fn new() -> Self {
        let mut gradient = Gradient::new();
        gradient.add_key(0.0, Color::BLUE.into());
        gradient.add_key(1.0, Color::RED.into());

        Self {
            gradient,
            precomputed_materials: vec![Handle::default(); 100],
        }
    }

    fn precompute_materials(&mut self, materials: &mut ResMut<Assets<ColorMaterial>>) {
        let num_precomputed_colors = 100;
        for i in 0..num_precomputed_colors {
            let gradient_point: Vec4 = self
                .gradient
                .sample(i as f32 / num_precomputed_colors as f32);
            let color = Color::rgba(
                gradient_point.x,
                gradient_point.y,
                gradient_point.z,
                gradient_point.w,
            );
            self.precomputed_materials[i] = materials.add(color.into());
        }
    }

    fn get_gradient_color_material(&self, ratio: &f32) -> Handle<ColorMaterial> {
        return self.precomputed_materials
            [(ratio.max(0.).min(1.) * (self.precomputed_materials.len() - 1) as f32) as usize]
            .clone();
    }
}

fn inspector_ui(
    world: &mut World,
    mut disabled: Local<bool>,
    mut last_time: Local<f32>,
    mut last_delta_t: Local<f32>,
) {
    let space_pressed = world.resource::<Input<KeyCode>>().just_pressed(KeyCode::H);
    if space_pressed {
        *disabled = !*disabled;
    }
    if *disabled {
        return;
    }

    let time = world.resource::<Time>().clone();
    if time.elapsed_seconds() - *last_time > 2. {
        *last_delta_t = time.delta_seconds();
        *last_time = time.elapsed_seconds();
    }
    let tps = if *last_delta_t > 0. {
        1. / *last_delta_t
    } else {
        0.
    };

    let mut egui_context = world
        .query_filtered::<&mut EguiContext, With<PrimaryWindow>>()
        .single(world)
        .clone();

    egui::Window::new("Config")
        .default_width(50.)
        .show(egui_context.get_mut(), |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                bevy_inspector_egui::bevy_inspector::ui_for_resource::<Config>(world, ui);

                ui.separator();
                ui.label(
                    RichText::new(format!(
                        "{}",
                        if tps > 300. {
                            "🚀"
                        } else if tps > 199. {
                            "👌"
                        } else if tps > 120. {
                            "🐢"
                        } else {
                            "🐌"
                        }
                    ))
                    .font(FontId::proportional(40.0)),
                );
                ui.label(format!("delta_t: {:.6} tps: {:.1}", *last_delta_t, tps,));

                ui.separator();
                ui.label("h: toggle ui");
                ui.label("i: reset config");
                ui.label("u: load config");
                ui.label("z: save config");
                ui.label("n: spawn 1 particle");
                ui.label("m: spawn 10 particles");
                ui.label("space: reset positions");
            });
        });
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
        // the background colors for all cameras
        .insert_resource(ClearColor(Color::rgb(0.0, 0.0, 0.0)))
        //.add_plugins(ResourceInspectorPlugin::<Config>::default())
        .insert_resource(config)
        .register_type::<Config>()
        .insert_resource(GradientResource::new())
        .add_systems(Startup, setup)
        .add_systems(Update, inspector_ui)
        .add_systems(
            Update,
            (
                keyboard_animation_control,
                gravity_system,
                calculate_density_system,
                pressure_force_system,
                color_system,
                move_system,
                bounce_system,
            )
                .chain(),
        )
        .run();
}

fn get_random_transform(window: &Window) -> Transform {
    let width_half = window.width() / 2.;
    let height_half = window.height() / 2.;

    let mut rng = thread_rng();
    let x = rng.gen_range(-width_half..width_half);
    let y = rng.gen_range(-height_half..height_half);
    Transform::from_translation(Vec3::new(x, y, 0.))
}

fn new_circle() -> shape::Circle {
    shape::Circle {
        radius: RADIUS,
        vertices: 4,
        ..default()
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut gradient_resource: ResMut<GradientResource>,
    windows: Query<&Window>,
    config: Res<Config>,
) {
    let window = windows.single();
    commands.spawn(Camera2dBundle::default());

    gradient_resource.precompute_materials(&mut materials);

    // spawn particles
    for _ in 0..config.num_particles {
        let velocity = Vec2::new(0., 0.);
        let transform = get_random_transform(&window);
        commands.spawn((
            MaterialMesh2dBundle {
                mesh: meshes.add(new_circle().into()).into(),
                material: materials.add(ColorMaterial::from(Color::PURPLE)),
                transform,
                ..default()
            },
            Velocity(velocity),
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

fn gravity_system(
    time: Res<Time>,
    config: Res<Config>,
    mut particles_query: Query<&mut Velocity, With<Particle>>,
) {
    if config.is_paused {
        return;
    }

    let delta_t = time.delta_seconds();
    if config.gravity == Vec2::ZERO {
        return;
    }

    particles_query.par_iter_mut().for_each(|mut velocity| {
        // print on direction change
        //if velocity.0.y > 0. && velocity.0.y + config.gravity.y * delta_t < 0.
        //    || velocity.0.y < 0. && velocity.0.y + config.gravity.y * delta_t > 0.
        //{
        //    println!("t translation: {:?}", transform.translation);
        //}
        velocity.0 += config.gravity * delta_t * 100.0;
    });
}

fn calculate_density_system(
    mut particles_query: Query<(&mut Density, &Transform), With<Density>>,
    particles_query2: Query<&Transform, With<Particle>>,
    config: Res<Config>,
) {
    if config.is_paused {
        return;
    }

    //let mut all_densities = Vec::new();
    //for (mut density, transform) in &mut particles_query {
    particles_query
        .par_iter_mut()
        .for_each(|(mut density, transform)| {
            let mut density_sum = 0.;
            let mut density_near_sum = 0.;
            for transform2 in &particles_query2 {
                if transform.translation == transform2.translation {
                    continue;
                }
                let distance = (transform2.translation - transform.translation).length();

                density_sum += spiky_kernel_pow_2(&config.smoothing_radius, &distance);
                density_near_sum += spiky_kernel_pow_3(&config.smoothing_radius, &distance);
            }
            density.far = density_sum;
            density.near = density_near_sum;

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
    mut particles_query: Query<(&mut Velocity, &Transform, &Density), With<Particle>>,
    particles_query2: Query<(&Transform, &Density), With<Particle>>,
    config: Res<Config>,
) {
    if config.is_paused {
        return;
    }
    let delta_t = time.delta_seconds();

    let pressure_from_density = |density: &f32| -> f32 {
        return (density - config.target_density) * config.pressure_multiplier;
    };

    let mut rng = thread_rng();
    let random_direction =
        Vec3::new(rng.gen_range(-1. ..1.), rng.gen_range(-1. ..1.), 0.).normalize();

    particles_query
        .par_iter_mut()
        .for_each(|(mut velocity, transform, density)| {
            let mut sum_pressure_force = Vec3::ZERO;
            for (transform2, density2) in &particles_query2 {
                // skip self
                if transform.translation == transform2.translation {
                    continue;
                }
                let offset = transform2.translation - transform.translation;
                let sqrt_dst = offset.length_squared();

                // skip if too far
                if sqrt_dst > config.smoothing_radius.powf(2.0) {
                    continue;
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
                //let shared_pressure_near = (pressure_from_density(&density.near)
                //    + pressure_from_density(&density2.near))
                //    * 0.5;

                sum_pressure_force += -direction
                    * derivative_spiky_pow_2(&config.smoothing_radius, &distance)
                    * shared_pressure
                    / density2.far.max(1.);

                //sum_pressure_force += -direction
                //    * derivative_spiky_pow_3(&config.smoothing_radius, &distance)
                //    * shared_pressure_near
                //    / density2.near.max(1.);
            }
            //let acceleration = sum_pressure_force / (density.far * density.near).max(1.);

            let acceleration = sum_pressure_force / density.far.max(1.);

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

// color particles based on their velocity
// color quads based on their density
fn color_system(
    config: Res<Config>,
    mut particles_query: Query<(&Velocity, &mut Handle<ColorMaterial>), With<Particle>>,
    mut quads_query: Query<(&Density, &mut Handle<ColorMaterial>), Without<Particle>>,
    gradient_resource: Res<GradientResource>,
) {
    if config.is_paused {
        return;
    }

    particles_query
        .par_iter_mut()
        .for_each(|(velocity, mut material)| {
            let speed_normalized = velocity.0.length() / 1000.0;
            //println!("speed_normalized {}", speed_normalized);
            *material = gradient_resource.get_gradient_color_material(&speed_normalized);
        });

    quads_query
        .par_iter_mut()
        .for_each(|(density, mut material)| {
            let density_normalized = density.far / config.target_density;
            *material = gradient_resource.get_gradient_color_material(&density_normalized);
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
}

//
//
//
//
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

// serialize animation to file
// including: struct Config
// excluding: particles
// format: json
// filename fluid-sim-isoTimestamp.json
fn save_config_to_file(config: Config) {
    let start_date = DateTime::from_timestamp(config.start_time, 0).unwrap();
    let mut file = File::create(format!(
        "./fluidsim/env-{}.json",
        start_date.to_rfc3339().replace(":", "_")
    ))
    .unwrap();

    // write config
    file.write_all(serde_json::to_string(&config).unwrap().as_bytes())
        .unwrap();
}

fn get_most_recent_file() -> Option<fs::DirEntry> {
    let mut files = fs::read_dir("./fluidsim/").unwrap();
    let mut most_recent_file = None;
    let mut most_recent_file_timestamp = 0;
    while let Some(file) = files.next() {
        let file = file.unwrap();
        let file_name = file.file_name().into_string().unwrap();
        if !file_name.starts_with("env-") {
            continue;
        }
        let file_timestamp = file_name
            .strip_prefix("env-")
            .unwrap()
            .strip_suffix(".json")
            .unwrap()
            .replace("_", ":")
            .parse::<DateTime<Utc>>()
            .unwrap()
            .timestamp();
        if file_timestamp > most_recent_file_timestamp {
            most_recent_file_timestamp = file_timestamp;
            most_recent_file = Some(file);
        }
    }
    most_recent_file
}

fn load_most_recent_config_from_file() -> Config {
    // make sure directory exists
    fs::create_dir_all("./fluidsim/").unwrap();

    let most_recent_file = get_most_recent_file();
    if most_recent_file.is_none() {
        println!("no config file found, using default");
        return Config::default();
    }
    let mut file = File::open(most_recent_file.unwrap().path()).unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();
    let loaded_config: Config = serde_json::from_str(&contents).unwrap();
    println!("loaded config: {:?}", loaded_config);

    let mut new_config = loaded_config.clone();
    new_config.start_time = Utc::now().timestamp();
    new_config
}

/*
fn debug_system(time: Res<Time>, mut last_time: Local<f32>) {
    // print delta_t
    if time.elapsed_seconds() - *last_time > (if *last_time == 0. { 2. } else { 10. }) {
        let delta_t = time.delta_seconds();
        println!("delta_t: {} tps: {}", delta_t, 1. / delta_t);
        *last_time = time.elapsed_seconds();
    }
}
*/
