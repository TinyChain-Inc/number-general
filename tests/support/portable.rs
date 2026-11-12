use super::{Float, NumberInstance, Trigonometry};

pub fn portable_real_kernel_vectors() {
    // libm 0.2.16 software-kernel bits. Representative values were checked against
    // the existing directed-rounding MPFR oracle (eight ULP accuracy bound).
    macro_rules! vector {
        ($method:ident, $input:expr, $f32:expr, $f64:expr) => {{
            let Float::F32(a) = Float::from($input as f32).$method().unwrap() else {
                panic!("f32 dtype")
            };
            let Float::F64(b) = Float::from($input as f64).$method().unwrap() else {
                panic!("f64 dtype")
            };
            assert_eq!(a.to_bits(), $f32, stringify!($method));
            assert_eq!(b.to_bits(), $f64, stringify!($method));
        }};
    }

    vector!(try_abs, -0.25, 0x3e800000, 0x3fd0000000000000);

    vector!(try_round, -0.25, 0x80000000, 0x8000000000000000);

    vector!(try_exp, 0.25, 0x3fa45af2, 0x3ff48b5e3c3e8186);

    vector!(try_ln, 2.0, 0x3f317218, 0x3fe62e42fefa39ef);

    vector!(try_sin, 0.25, 0x3e7d5777, 0x3fcfaaeed4f31577);

    vector!(try_cos, 0.25, 0x3f780aa5, 0x3fef01549f7deea1);

    vector!(try_tan, 0.25, 0x3e82bc2d, 0x3fd05785a43c4c56);

    vector!(try_sinh, 0.25, 0x3e815667, 0x3fd02accd9d08102);

    vector!(try_cosh, 0.25, 0x3f840558, 0x3ff080ab05ca6146);

    vector!(try_tanh, 0.25, 0x3e7acbf5, 0x3fcf597ea69a1c86);

    vector!(try_asin, 0.25, 0x3e815f4e, 0x3fd02be9ce0b87cd);

    vector!(try_acos, 0.25, 0x3fa8b807, 0x3ff51700e0c14b25);

    vector!(try_atan, 0.25, 0x3e7adbb0, 0x3fcf5b75f92c80dd);

    vector!(try_asinh, 0.25, 0x3e7d67da, 0x3fcfacfb2399e637);

    vector!(try_acosh, 2.0, 0x3fa89214, 0x3ff5124271980434);

    vector!(try_atanh, 0.25, 0x3e82c578, 0x3fd058aefa811452);

    let Float::F32(tiny) = Float::from(-103f32).try_exp().unwrap() else {
        panic!("dtype")
    };
    assert_eq!(tiny.to_bits(), 1);
    let Float::F64(tiny) = Float::from(-745f64).try_exp().unwrap() else {
        panic!("dtype")
    };
    assert_eq!(tiny.to_bits(), 1);
    for value in [0.0f64, -0.0] {
        let Float::F64(sine) = Float::from(value).try_sin().unwrap() else {
            panic!("dtype")
        };
        assert_eq!(sine.to_bits(), value.to_bits());
        let Float::F64(rounded) = Float::from(value).try_round().unwrap() else {
            panic!("dtype")
        };
        assert_eq!(rounded.to_bits(), value.to_bits());
    }
    for value in [f32::MIN_POSITIVE * f32::EPSILON, f32::MIN_POSITIVE] {
        assert_eq!(
            Float::from(value).try_mul(1f32.into()).unwrap(),
            Float::from(value)
        );
    }
    assert!(matches!(Float::from(f64::NAN).try_ln().unwrap(), Float::F64(v) if v.is_nan()));
    assert_eq!(
        Float::from(f64::INFINITY).try_exp().unwrap(),
        Float::from(f64::INFINITY)
    );
}
