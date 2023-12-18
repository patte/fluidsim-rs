# Fluidsim

Make particles go splish splash.
Inspired by [Coding Adventure: Simulating Fluids](https://www.youtube.com/watch?v=rSKMYc1CQHE)

## run
```bash
cargo run --release
```

## develop
```bash
cargo watch -x "run --release" --ignore '*.json'
```

## wasm
```bash
cargo build --profile wasm-release --target wasm32-unknown-unknown
wasm-bindgen --no-typescript --target web \
    --out-dir ./out/ \
    --out-name "fluidsim" \
    ./target/wasm32-unknown-unknown/release/fluidsim.wasm

cargo watch --ignore '*.json' -s "cargo build --profile wasm-release --target wasm32-unknown-unknown && wasm-bindgen --no-typescript --target web --out-dir ./out/ --out-name \"fluidsim\" ./target/wasm32-unknown-unknown/release/fluidsim.wasm"
```