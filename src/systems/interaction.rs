use bevy::{input::touch::TouchPhase, prelude::*, sprite::MaterialMesh2dBundle};

use crate::{
    get_position_in_grid, get_random_transform, load_most_recent_config_from_file, new_circle,
    save_config_to_file, Config, Density, InteractionInputs, Measurements, Particle,
    PredictedPosition, Velocity, CIRCLE_RATIO, MASS, SCALE_FACTOR,
};
pub fn keyboard_interaction_system(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
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
    if keyboard_input.just_pressed(KeyCode::KeyP) {
        config.is_paused = !config.is_paused;
        key_pressed = true;
    }

    // pause after next frame
    if keyboard_input.just_pressed(KeyCode::ArrowRight) {
        config.is_paused = false;
        config.pause_after_next_frame = true;
        key_pressed = true;
    }

    // reset config
    if keyboard_input.just_pressed(KeyCode::KeyI) {
        *config = Config::default();

        key_pressed = true;
    }

    // save config
    if keyboard_input.just_pressed(KeyCode::KeyZ) {
        save_config_to_file(config.clone());
        println!("config saved!");
        key_pressed = true;
    }

    //load config
    if keyboard_input.just_pressed(KeyCode::KeyU) {
        *config = load_most_recent_config_from_file();
        println!("config loaded!");
        key_pressed = true;
    }

    // toggle auto save
    if keyboard_input.just_pressed(KeyCode::KeyK) {
        config.auto_save = !config.auto_save;
        println!("auto_save: {}", config.auto_save);
        key_pressed = true;
    }

    // toggle cccb display sender
    if keyboard_input.just_pressed(KeyCode::KeyC) {
        config.cccb_display_sender = !config.cccb_display_sender;
        println!("cccb_display_sender: {}", config.cccb_display_sender);
        key_pressed = true;
    }

    // pop new particle at random position
    if keyboard_input.just_pressed(KeyCode::KeyN) || keyboard_input.just_pressed(KeyCode::KeyM) {
        let spawn_num_particles = if keyboard_input.just_pressed(KeyCode::KeyN) {
            1
        } else {
            10
        };
        for _ in 0..spawn_num_particles {
            commands.spawn((
                MaterialMesh2dBundle {
                    mesh: meshes
                        .add(new_circle(config.smoothing_radius * CIRCLE_RATIO))
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

    // exit
    if keyboard_input.just_pressed(KeyCode::KeyQ) {
        std::process::exit(0);
    }
}

pub fn mouse_interaction_system(
    q_windows: Query<&Window, With<bevy_internal::window::PrimaryWindow>>,
    buttons: Res<ButtonInput<MouseButton>>,
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

pub fn touch_interaction_system(
    q_windows: Query<&Window, With<bevy_internal::window::PrimaryWindow>>,
    mut touch_evr: EventReader<TouchInput>,
    mut interaction_inputs: ResMut<InteractionInputs>,
    config: Res<Config>,
) {
    let window = q_windows.single();

    for touch in touch_evr.read() {
        if touch.phase == TouchPhase::Started || touch.phase == TouchPhase::Moved {
            let position = touch.position;
            let x = (position.x - (window.width() / 2.)) * SCALE_FACTOR;
            let y = -(position.y - (window.height() / 2.)) * SCALE_FACTOR;
            let interaction_pos = Vec2::new(x, y);
            interaction_inputs.point = Some(interaction_pos);
            interaction_inputs.strength = config.interaction_input_strength;
        } else {
            interaction_inputs.point = None;
            interaction_inputs.strength = 0.;
        }
    }
}
