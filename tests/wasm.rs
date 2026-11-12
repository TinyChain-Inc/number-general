#![cfg(all(target_arch = "wasm32", target_os = "unknown"))]

use number_general::{ErrorKind, Float, Number, NumberInstance, Trigonometry};
use wasm_bindgen_test::wasm_bindgen_test;

#[path = "support/portable.rs"]
mod portable;

#[wasm_bindgen_test]
fn checked_portability_vectors() {
    portable::portable_real_kernel_vectors();
}

#[wasm_bindgen_test]
fn checked_overflow_is_structured() {
    let result = Number::from(u8::MAX).try_add(Number::from(1u8));
    assert_eq!(result.unwrap_err().kind(), ErrorKind::Overflow);
    assert_eq!(
        Number::from(7i32).try_add(Number::from(8i32)).unwrap(),
        15i32.into()
    );
}
