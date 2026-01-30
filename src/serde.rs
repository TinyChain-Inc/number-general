use std::fmt;

use serde::de::{self, Deserialize, Deserializer, Visitor};
use serde::ser::{Serialize, SerializeSeq, Serializer};

use super::{
    Boolean, Complex, Float, Int, Number, NumberVisitor, UInt, _Complex, ERR_COMPLEX, ERR_NUMBER,
};

impl Serialize for Boolean {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_bool(bool::from(self))
    }
}

impl<'de> Deserialize<'de> for Boolean {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        bool::deserialize(deserializer).map(Self::from)
    }
}

impl<'de> Deserialize<'de> for Complex {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let [re, im]: [f64; 2] = Deserialize::deserialize(deserializer)?;
        Ok(num::Complex::new(re, im).into())
    }
}

impl Serialize for Complex {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Complex::C32(c) => {
                let mut seq = s.serialize_seq(Some(2))?;
                seq.serialize_element(&c.re)?;
                seq.serialize_element(&c.im)?;
                seq.end()
            }
            Complex::C64(c) => {
                let mut seq = s.serialize_seq(Some(2))?;
                seq.serialize_element(&c.re)?;
                seq.serialize_element(&c.im)?;
                seq.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for Float {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        f64::deserialize(deserializer).map(Self::from)
    }
}

impl Serialize for Float {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Float::F32(f) => s.serialize_f32(*f),
            Float::F64(f) => s.serialize_f64(*f),
        }
    }
}

impl<'de> Deserialize<'de> for Int {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        i64::deserialize(deserializer).map(Self::from)
    }
}

impl Serialize for Int {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Int::I8(i) => s.serialize_i8(*i),
            Int::I16(i) => s.serialize_i16(*i),
            Int::I32(i) => s.serialize_i32(*i),
            Int::I64(i) => s.serialize_i64(*i),
        }
    }
}

impl<'de> Deserialize<'de> for UInt {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        u64::deserialize(deserializer).map(Self::from)
    }
}

impl Serialize for UInt {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            UInt::U8(u) => s.serialize_u8(*u),
            UInt::U16(u) => s.serialize_u16(*u),
            UInt::U32(u) => s.serialize_u32(*u),
            UInt::U64(u) => s.serialize_u64(*u),
        }
    }
}

impl<'de> Visitor<'de> for NumberVisitor {
    type Value = Number;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(ERR_NUMBER)
    }

    #[inline]
    fn visit_bool<E: de::Error>(self, b: bool) -> Result<Self::Value, E> {
        self.bool(b)
    }

    #[inline]
    fn visit_i8<E: de::Error>(self, i: i8) -> Result<Self::Value, E> {
        self.i8(i)
    }

    #[inline]
    fn visit_i16<E: de::Error>(self, i: i16) -> Result<Self::Value, E> {
        self.i16(i)
    }

    #[inline]
    fn visit_i32<E: de::Error>(self, i: i32) -> Result<Self::Value, E> {
        self.i32(i)
    }

    #[inline]
    fn visit_i64<E: de::Error>(self, i: i64) -> Result<Self::Value, E> {
        self.i64(i)
    }

    #[inline]
    fn visit_u8<E: de::Error>(self, u: u8) -> Result<Self::Value, E> {
        self.u8(u)
    }

    #[inline]
    fn visit_u16<E: de::Error>(self, u: u16) -> Result<Self::Value, E> {
        self.u16(u)
    }

    #[inline]
    fn visit_u32<E: de::Error>(self, u: u32) -> Result<Self::Value, E> {
        self.u32(u)
    }

    #[inline]
    fn visit_u64<E>(self, u: u64) -> Result<Self::Value, E> {
        self.u64(u)
    }

    #[inline]
    fn visit_f32<E>(self, f: f32) -> Result<Self::Value, E> {
        self.f32(f)
    }

    #[inline]
    fn visit_f64<E>(self, f: f64) -> Result<Self::Value, E> {
        self.f64(f)
    }

    #[inline]
    fn visit_str<E: de::Error>(self, s: &str) -> Result<Self::Value, E> {
        s.parse().map_err(serde::de::Error::custom)
    }

    #[inline]
    fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let re = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(0, &ERR_COMPLEX))?;

        let im = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(1, &ERR_COMPLEX))?;

        Ok(Number::Complex(Complex::C64(_Complex::<f64>::new(re, im))))
    }
}

impl<'de> Deserialize<'de> for Number {
    fn deserialize<D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Self, <D as Deserializer<'de>>::Error> {
        deserializer.deserialize_any(NumberVisitor)
    }
}

impl Serialize for Number {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Number::Bool(b) => b.serialize(s),
            Number::Complex(c) => c.serialize(s),
            Number::Float(f) => f.serialize(s),
            Number::Int(i) => i.serialize(s),
            Number::UInt(u) => u.serialize(s),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::approx_constant)] // tests use explicit literals to validate serialization
    fn test_serialize() {
        let numbers = [
            Number::from(false),
            Number::from(12u16),
            Number::from(-3),
            Number::from(3.14),
            Number::from(_Complex::<f32>::new(0., -1.414)),
        ];

        for expected in &numbers {
            let serialized = serde_json::to_string(expected).unwrap();
            let actual = serde_json::from_str(&serialized).unwrap();

            assert_eq!(expected, &actual);
        }
    }
}
