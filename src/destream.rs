use destream::de::{self, Decoder, FromStream, Visitor};
use destream::en::{EncodeSeq, Encoder, IntoStream, ToStream};
use futures::TryFutureExt;

use super::{
    Boolean, Complex, Float, Int, Number, NumberVisitor, UInt, _Complex, ERR_COMPLEX, ERR_NUMBER,
};

impl FromStream for Boolean {
    type Context = ();

    async fn from_stream<D: Decoder>(
        cxt: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        bool::from_stream(cxt, decoder).map_ok(Self::from).await
    }
}

impl<'en> ToStream<'en> for Boolean {
    fn to_stream<E: Encoder<'en>>(&'en self, e: E) -> Result<E::Ok, E::Error> {
        e.encode_bool(bool::from(self))
    }
}

impl<'en> IntoStream<'en> for Boolean {
    fn into_stream<E: Encoder<'en>>(self, e: E) -> Result<E::Ok, E::Error> {
        e.encode_bool(bool::from(self))
    }
}

impl FromStream for Complex {
    type Context = ();

    async fn from_stream<D: Decoder>(
        cxt: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        let [re, im]: [f64; 2] = <[f64; 2] as FromStream>::from_stream(cxt, decoder).await?;
        Ok(num::Complex::new(re, im).into())
    }
}

impl<'en> ToStream<'en> for Complex {
    fn to_stream<E: Encoder<'en>>(&'en self, e: E) -> Result<E::Ok, E::Error> {
        match self {
            Complex::C32(c) => {
                let mut seq = e.encode_seq(Some(2))?;
                seq.encode_element(&c.re)?;
                seq.encode_element(&c.im)?;
                seq.end()
            }
            Complex::C64(c) => {
                let mut seq = e.encode_seq(Some(2))?;
                seq.encode_element(&c.re)?;
                seq.encode_element(&c.im)?;
                seq.end()
            }
        }
    }
}

impl<'en> IntoStream<'en> for Complex {
    fn into_stream<E: Encoder<'en>>(self, e: E) -> Result<E::Ok, E::Error> {
        match self {
            Complex::C32(c) => {
                let mut seq = e.encode_seq(Some(2))?;
                seq.encode_element(c.re)?;
                seq.encode_element(c.im)?;
                seq.end()
            }
            Complex::C64(c) => {
                let mut seq = e.encode_seq(Some(2))?;
                seq.encode_element(c.re)?;
                seq.encode_element(c.im)?;
                seq.end()
            }
        }
    }
}

impl FromStream for Float {
    type Context = ();

    async fn from_stream<D: Decoder>(
        cxt: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        f64::from_stream(cxt, decoder).map_ok(Self::from).await
    }
}

impl<'en> ToStream<'en> for Float {
    fn to_stream<E: Encoder<'en>>(&'en self, e: E) -> Result<E::Ok, E::Error> {
        match *self {
            Float::F32(f) => e.encode_f32(f),
            Float::F64(f) => e.encode_f64(f),
        }
    }
}

impl<'en> IntoStream<'en> for Float {
    fn into_stream<E: Encoder<'en>>(self, e: E) -> Result<E::Ok, E::Error> {
        match self {
            Float::F32(f) => e.encode_f32(f),
            Float::F64(f) => e.encode_f64(f),
        }
    }
}

impl FromStream for Int {
    type Context = ();

    async fn from_stream<D: Decoder>(
        cxt: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        i64::from_stream(cxt, decoder).map_ok(Self::from).await
    }
}

impl<'en> ToStream<'en> for Int {
    fn to_stream<E: Encoder<'en>>(&'en self, e: E) -> Result<E::Ok, E::Error> {
        match *self {
            Int::I8(i) => e.encode_i8(i),
            Int::I16(i) => e.encode_i16(i),
            Int::I32(i) => e.encode_i32(i),
            Int::I64(i) => e.encode_i64(i),
        }
    }
}

impl<'en> IntoStream<'en> for Int {
    fn into_stream<E: Encoder<'en>>(self, e: E) -> Result<E::Ok, E::Error> {
        match self {
            Int::I8(i) => e.encode_i8(i),
            Int::I16(i) => e.encode_i16(i),
            Int::I32(i) => e.encode_i32(i),
            Int::I64(i) => e.encode_i64(i),
        }
    }
}

impl FromStream for UInt {
    type Context = ();

    async fn from_stream<D: Decoder>(
        cxt: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        u64::from_stream(cxt, decoder).map_ok(Self::from).await
    }
}

impl<'en> ToStream<'en> for UInt {
    fn to_stream<E: Encoder<'en>>(&'en self, e: E) -> Result<E::Ok, E::Error> {
        match *self {
            UInt::U8(u) => e.encode_u8(u),
            UInt::U16(u) => e.encode_u16(u),
            UInt::U32(u) => e.encode_u32(u),
            UInt::U64(u) => e.encode_u64(u),
        }
    }
}

impl<'en> IntoStream<'en> for UInt {
    fn into_stream<E: Encoder<'en>>(self, e: E) -> Result<E::Ok, E::Error> {
        match self {
            UInt::U8(u) => e.encode_u8(u),
            UInt::U16(u) => e.encode_u16(u),
            UInt::U32(u) => e.encode_u32(u),
            UInt::U64(u) => e.encode_u64(u),
        }
    }
}

impl Visitor for NumberVisitor {
    type Value = Number;

    fn expecting() -> &'static str {
        ERR_NUMBER
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
    fn visit_u64<E: de::Error>(self, u: u64) -> Result<Self::Value, E> {
        self.u64(u)
    }

    #[inline]
    fn visit_f32<E: de::Error>(self, f: f32) -> Result<Self::Value, E> {
        self.f32(f)
    }

    #[inline]
    fn visit_f64<E: de::Error>(self, f: f64) -> Result<Self::Value, E> {
        self.f64(f)
    }

    #[inline]
    fn visit_string<E: de::Error>(self, s: String) -> Result<Self::Value, E> {
        s.parse().map_err(destream::de::Error::custom)
    }

    async fn visit_seq<A: destream::de::SeqAccess>(
        self,
        mut seq: A,
    ) -> Result<Self::Value, A::Error> {
        let re = seq
            .next_element(())
            .await?
            .ok_or_else(|| de::Error::invalid_length(0, ERR_COMPLEX))?;

        let im = seq
            .next_element(())
            .await?
            .ok_or_else(|| de::Error::invalid_length(1, ERR_COMPLEX))?;

        Ok(Number::Complex(Complex::C64(_Complex::<f64>::new(re, im))))
    }
}

impl FromStream for Number {
    type Context = ();

    async fn from_stream<D: Decoder>(
        _context: (),
        decoder: &mut D,
    ) -> Result<Self, <D as Decoder>::Error> {
        decoder.decode_any(NumberVisitor).await
    }
}

impl<'en> ToStream<'en> for Number {
    fn to_stream<E: destream::Encoder<'en>>(&'en self, e: E) -> Result<E::Ok, E::Error> {
        match self {
            Number::Bool(b) => b.to_stream(e),
            Number::Complex(c) => c.to_stream(e),
            Number::Float(f) => f.to_stream(e),
            Number::Int(i) => i.to_stream(e),
            Number::UInt(u) => u.to_stream(e),
        }
    }
}

impl<'en> IntoStream<'en> for Number {
    fn into_stream<E: destream::Encoder<'en>>(self, e: E) -> Result<E::Ok, E::Error> {
        match self {
            Number::Bool(b) => b.into_stream(e),
            Number::Complex(c) => c.into_stream(e),
            Number::Float(f) => f.into_stream(e),
            Number::Int(i) => i.into_stream(e),
            Number::UInt(u) => u.into_stream(e),
        }
    }
}

// Tests for this module are implemented as part of the "value" feature of the destream_json crate
