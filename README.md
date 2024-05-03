# Fluidsim

Make particles go splish splash.
Inspired by [Coding Adventure: Simulating Fluids](https://www.youtube.com/watch?v=rSKMYc1CQHE)

Written in Rust in the Bevy engine's ECS system, which makes sheduling easy with the cost of simplicity.

## native

### run
```bash
cargo run --release
```

### develop
```bash
cargo watch -x "run --release" --ignore '*.json'
```

## wasm

### prepare
```bash
rustup target add wasm32-unknown-unknown
## make sure to get update wasm-bindgen binary
cargo install -f wasm-bindgen-cli
```

### run
```bash
cargo build --profile wasm-release --target wasm32-unknown-unknown
wasm-bindgen --no-typescript --target web \
    --out-dir ./out/ \
    --out-name "fluidsim" \
    ./target/wasm32-unknown-unknown/wasm-release/fluidsim.wasm
```

### serve
```bash
dufs -p 3000 ./out/
```
<small>https://github.com/sigoden/dufs</small>

### develop
```bash
cargo watch --ignore '*.json' -s "cargo build --profile wasm-release --target wasm32-unknown-unknown && wasm-bindgen --no-typescript --target web --out-dir ./out/ --out-name \"fluidsim\" ./target/wasm32-unknown-unknown/wasm-release/fluidsim.wasm"
```