use crate::Config;
use bevy::prelude::*;
use bevy::render::view::screenshot::ScreenshotManager;
use bevy::window::PrimaryWindow;
use cccb_display::{CccbDisplayImagePackage, CccbImageSender};
use image::GenericImageView;

pub fn cccb_display_system(
    main_window: Query<Entity, With<PrimaryWindow>>,
    mut screenshot_manager: ResMut<ScreenshotManager>,
    mut elapsed: Local<f32>,
    time: Res<Time>,
    config: Res<Config>,
) {
    if !config.cccb_display_sender {
        return;
    }

    *elapsed += time.delta_seconds();

    // max pps 60
    if *elapsed < 1. / 60. {
        return;
    }
    *elapsed = 0.;

    let _ = screenshot_manager.take_screenshot(main_window.single(), move |img| {
        match img.try_into_dynamic() {
            Ok(mut dynamic_img) => {
                if dynamic_img.width() == 0 || dynamic_img.height() == 0 {
                    println!(
                        "Screenshot empty: {}x{}",
                        dynamic_img.width(),
                        dynamic_img.height()
                    );
                    return;
                }

                // resize image /2
                let (width, height) = dynamic_img.dimensions();
                dynamic_img =
                    dynamic_img.resize(width / 2, height / 2, image::imageops::FilterType::Nearest);

                // crop image
                let (width, height) = dynamic_img.dimensions();
                let crop_x = (width - CccbDisplayImagePackage::WIDTH as u32 * 8) / 2;
                let crop_y = (height - CccbDisplayImagePackage::HEIGHT as u32) / 2;
                dynamic_img = dynamic_img.crop_imm(
                    crop_x,
                    crop_y,
                    CccbDisplayImagePackage::WIDTH as u32 * 8,
                    CccbDisplayImagePackage::HEIGHT as u32,
                );

                fn pix_is_on(pix: &image::Rgba<u8>) -> bool {
                    pix[0] > 1 || pix[1] > 1 || pix[2] > 1
                }
                let img_packed = CccbDisplayImagePackage::new(dynamic_img, pix_is_on, false);

                // save screenshot to disk
                img_packed
                    .to_luma8()
                    .save("./screenshots/screenshot.png")
                    .unwrap();

                let mut sender = CccbImageSender::new_from_env();
                sender.send_package(&img_packed);
            }
            Err(e) => {
                println!("Error: {}", e);
            }
        }
    });
}
