use super::Config;
use std::{
    fs::{self, File},
    io::{Read, Write},
};

use chrono::prelude::{DateTime, Utc};

use serde_json;

// serialize config to file
// format: json
// filename env-isoTimestamp.json
pub fn save_config_to_file(config: Config) {
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

pub fn get_most_recent_file() -> Option<fs::DirEntry> {
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

pub fn load_most_recent_config_from_file() -> Config {
    // no files in wasm
    if cfg!(target_arch = "wasm32") {
        return Config::default();
    }
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
