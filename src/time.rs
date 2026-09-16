use rsmpeg::avutil;
use rsmpeg::ffi;
use rsmpeg::ffi::AVRational;

use std::time::Duration;

/// Represents a time or duration.
///
/// [`Time`] may represent a PTS (presentation timestamp), DTS (decoder timestamp) or a duration,
/// depending on the function that returns it.
///
/// [`Time`] may represent a non-existing time, in which case [`Time::has_value`] will return
/// `false`, and conversions to seconds will return `0.0`. FFmpeg's `AV_NOPTS_VALUE`
/// sentinel means exactly that, so it is folded into "no value" at the FFI
/// boundary (see [`Time::new`]) and by every accessor below.
///
/// Equality and ordering are expressed in **seconds**, not in `(time, time_base)`
/// pairs: two stamps that denote the same instant are equal even if their time
/// bases differ (see [`PartialEq`]).
///
/// A [`Time`] object may be aligned with another [`Time`] object, which produces an [`Aligned`]
/// object, on which arithmetic operations can be performed.
#[derive(Debug, Copy, Clone)]
pub struct Time {
    pub time: Option<i64>,
    pub time_base: AVRational,
}

impl Time {
    /// Create a new time by its time value and time base in which the time is expressed.
    ///
    /// `AV_NOPTS_VALUE` (FFmpeg's "no timestamp" sentinel, which streams and
    /// frames routinely carry) is normalised to `None` here — this is the FFI
    /// boundary, so callers can trust [`Time::has_value`] afterwards instead of
    /// re-checking the sentinel themselves.
    ///
    /// # Arguments
    ///
    /// * `time` - Relative time in `time_base` units.
    /// * `time_base` - Time base of source.
    pub fn new(time: Option<i64>, time_base: AVRational) -> Time {
        Self {
            time: time.filter(|time| *time != ffi::AV_NOPTS_VALUE),
            time_base,
        }
    }

    /// Creates a new timestamp that reprsents `nth` of a second.
    ///
    /// # Arguments
    ///
    /// * `nth` - Denominator of the time in seconds as in `1 / nth`.
    pub fn from_nth_of_a_second(nth: usize) -> Self {
        Self {
            time: Some(1),
            time_base: new_rational(1, nth as i32),
        }
    }

    /// Creates a new timestamp from a number of seconds.
    ///
    /// # Arguments
    ///
    /// * `secs` - Number of seconds.
    pub fn from_secs(secs: f32) -> Self {
        Self {
            time: Some((secs * TIME_BASE.den as f32).round() as i64),
            time_base: TIME_BASE,
        }
    }

    /// Creates a new timestamp from a number of seconds.
    ///
    /// # Arguments
    ///
    /// * `secs` - Number of seconds.
    pub fn from_secs_f64(secs: f64) -> Self {
        Self {
            time: Some((secs * TIME_BASE.den as f64).round() as i64),
            time_base: TIME_BASE,
        }
    }

    /// Creates a new timestamp with `time` time units, each represents one / `base_den` seconds.
    ///
    /// # Arguments
    ///
    /// * `time` - Relative time in `time_base` units.
    /// * `base_den` - Time base denominator i.e. time base is `1 / base_den`.
    pub fn from_units(time: usize, base_den: usize) -> Self {
        Self {
            time: Some(time as i64),
            time_base: new_rational(1, base_den as i32),
        }
    }

    /// Create a new zero-valued timestamp.
    ///
    /// 时间基取 [`TIME_BASE`]，因此与 `Time::new(Some(0), TIME_BASE)`（以及
    /// `Time::from_secs(0.0)`）完全相等 —— 早先这里硬编码 `1/90000`，会出现
    /// "同为 0 秒却不相等" 的荒谬结果。
    pub fn zero() -> Self {
        Time {
            time: Some(0),
            time_base: TIME_BASE,
        }
    }

    /// Whether the [`Time`] carries a usable value at all.
    ///
    /// Both "no time" spellings report `false`: no value whatsoever (`None`), and
    /// the `AV_NOPTS_VALUE` sentinel FFmpeg writes when a stream simply has no
    /// timestamp. This is the predicate to branch on before converting to seconds
    /// — converting a NOPTS would yield ≈ -9.2e13 seconds.
    /// [`Self::into_value`], the seconds conversions and the comparisons all
    /// agree with it.
    pub fn has_value(&self) -> bool {
        self.value().is_some()
    }

    /// The raw value, with the `AV_NOPTS_VALUE` sentinel filtered out.
    ///
    /// [`Time::new`] already normalises the sentinel, but `time` is a public
    /// field, so a hand-built `Time { time: Some(AV_NOPTS_VALUE), .. }` is still
    /// possible; every accessor funnels through here so that there is only one
    /// notion of "has a value".
    fn value(&self) -> Option<i64> {
        self.time.filter(|time| *time != ffi::AV_NOPTS_VALUE)
    }

    /// The instant in seconds, or `None` when there is nothing to convert: no
    /// value at all, or a degenerate time base (`0/0`).
    ///
    /// The comparison and formatting impls key off this, so "equal", "ordered"
    /// and "printed" always refer to the same number.
    fn seconds_or_none(&self) -> Option<f64> {
        let time = self.value()?;
        if self.time_base.num == 0 || self.time_base.den == 0 {
            return None;
        }
        Some(time as f64 * (self.time_base.num as f64 / self.time_base.den as f64))
    }

    /// Align the timestamp with another timestamp, which will convert the `rhs` timestamp to the
    /// same time base, such that operations can be performed upon the aligned timestamps.
    ///
    /// # Arguments
    ///
    /// * `rhs` - Right-hand side timestamp.
    ///
    /// # Return value
    ///
    /// Two timestamps that are aligned.
    pub fn aligned_with(&self, rhs: Time) -> Aligned {
        Aligned {
            lhs: self.value(),
            rhs: rhs
                .value()
                .map(|rhs_time| rhs_time.rescale(rhs.time_base, self.time_base)),
            time_base: self.time_base,
        }
    }

    /// Get number of seconds as floating point value.
    ///
    /// Single-precision on purpose (the historical API); it is computed from
    /// [`Self::as_secs_f64`] solely so the two can never disagree. Use the `f64`
    /// variant when precision matters.
    pub fn as_secs(&self) -> f32 {
        self.as_secs_f64() as f32
    }

    /// Get number of seconds as floating point value.
    ///
    /// Returns `0.0` when there is no usable value ([`Self::has_value`] is
    /// `false`) or the time base is degenerate (`0/0`), which also keeps the
    /// result finite — a NOPTS would otherwise turn into ≈ -9.2e13 seconds.
    pub fn as_secs_f64(&self) -> f64 {
        self.seconds_or_none().unwrap_or(0.0)
    }

    /// Convert to underlying time to `i64` (the number of time units).
    ///
    /// Returns `None` when there is no usable value — including the
    /// `AV_NOPTS_VALUE` sentinel — so it agrees with [`Self::has_value`].
    ///
    /// Assumes that the caller knows the time base and applies it correctly when doing arithmetic
    /// operations on the time value.
    pub fn into_value(self) -> Option<i64> {
        self.value()
    }

    /// Align the timestamp along another `time_base`.
    ///
    /// # Arguments
    ///
    /// * `time_base` - Target time base.
    pub fn aligned_with_rational(&self, time_base: AVRational) -> Time {
        Time {
            time: self
                .value()
                .map(|time| time.rescale(self.time_base, time_base)),
            time_base,
        }
    }
}

/////////////////////////////////
/////////////////////////////////

pub const TIME_BASE: AVRational = avutil::ra(ffi::AV_TIME_BASE_Q.num, ffi::AV_TIME_BASE_Q.den);

pub trait Rescale {
    fn rescale<S, D>(&self, source: S, destination: D) -> i64
    where
        S: Into<AVRational>,
        D: Into<AVRational>;

    fn rescale_with<S, D>(&self, source: S, destination: D, rounding: ffi::AVRounding) -> i64
    where
        S: Into<AVRational>,
        D: Into<AVRational>;
}

impl<T: Into<i64> + Clone> Rescale for T {
    fn rescale<S, D>(&self, source: S, destination: D) -> i64
    where
        S: Into<AVRational>,
        D: Into<AVRational>,
    {
        avutil::av_rescale_q(self.clone().into(), source.into(), destination.into())
    }

    fn rescale_with<S, D>(&self, source: S, destination: D, rounding: ffi::AVRounding) -> i64
    where
        S: Into<AVRational>,
        D: Into<AVRational>,
    {
        avutil::av_rescale_q_rnd(
            self.clone().into(),
            source.into(),
            destination.into(),
            rounding as _,
        )
    }
}

#[inline(always)]
pub fn new_rational(num: i32, den: i32) -> AVRational {
    avutil::ra(num, den)
}

#[inline(always)]
pub fn av_rational_eq(a: &AVRational, b: &AVRational) -> bool {
    a.num == b.num && a.den == b.den
}

impl PartialEq for Time {
    /// Compares the instants, **in seconds**, not the `(time, time_base)` pairs:
    /// `Time::from_units(1, 4)` and `Time::new(Some(2), new_rational(1, 2))` both
    /// mean 0.5 s and are therefore equal.
    ///
    /// Requiring the raw fields to match used to make "the same instant" compare
    /// unequal whenever the time bases differed. Two "no value" times are equal
    /// (`None == None`); a "no value" time never equals a valued one. The result
    /// matches [`PartialOrd`]: `a == b` ⟺ `a.partial_cmp(&b) == Some(Equal)`.
    fn eq(&self, other: &Self) -> bool {
        self.seconds_or_none() == other.seconds_or_none()
    }
}

impl Eq for Time {}

impl PartialOrd for Time {
    /// Orders by seconds, "no value" first; [`PartialEq`] uses the same key, so
    /// the `PartialOrd` contract holds. Always `Some`, since the key is always
    /// comparable.
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(match (self.seconds_or_none(), other.seconds_or_none()) {
            (None, None) => std::cmp::Ordering::Equal,
            (None, Some(_)) => std::cmp::Ordering::Less,
            (Some(_), None) => std::cmp::Ordering::Greater,
            // `den != 0` and a finite `time` cannot produce NaN; the fallback
            // just keeps the ordering total if that ever changes.
            (Some(lhs), Some(rhs)) => lhs.partial_cmp(&rhs).unwrap_or(std::cmp::Ordering::Equal),
        })
    }
}

impl From<Duration> for Time {
    /// Convert from a [`Duration`] to [`Time`].
    #[inline]
    fn from(duration: Duration) -> Self {
        Time::from_secs_f64(duration.as_secs_f64())
    }
}

impl From<Time> for Duration {
    /// Convert from a [`Time`] to a Rust-native [`Duration`].
    fn from(timestamp: Time) -> Self {
        Duration::from_secs_f64(timestamp.as_secs_f64().max(0.0))
    }
}

impl std::fmt::Display for Time {
    /// Format [`Time`] as follows:
    ///
    /// * If the inner value is usable: the number of seconds, e.g. `0.5 secs`.
    /// * Otherwise: `none`.
    ///
    /// Printing the seconds (from [`Time::as_secs_f64`]) rather than the raw
    /// `time * time_base.num` numerator keeps `Display` panic-free: that product
    /// overflows `i64` for large timestamps and would panic in debug builds.
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.seconds_or_none() {
            Some(secs) => write!(f, "{secs} secs"),
            None => write!(f, "none"),
        }
    }
}

/// This is a virtual object that represents two aligned times.
///
/// On this object, arthmetic operations can be performed that operate on the two contained times.
/// This virtual object ensures that the interface to these operations is safe.
#[derive(Debug, Clone)]
pub struct Aligned {
    lhs: Option<i64>,
    rhs: Option<i64>,
    time_base: AVRational,
}

impl Aligned {
    /// Add two timestamps together.
    pub fn add(self) -> Time {
        self.apply(|lhs, rhs| lhs + rhs)
    }

    /// Subtract the right-hand side timestamp from the left-hand side timestamp.
    pub fn subtract(self) -> Time {
        self.apply(|lhs, rhs| lhs - rhs)
    }

    /// Apply operation `f` on aligned timestamps.
    ///
    /// The closure operates on the numerator of two aligned times.
    ///
    /// # Arguments
    ///
    /// * `f` - Function to apply on the two aligned time numerator values.
    fn apply<F>(self, f: F) -> Time
    where
        F: FnOnce(i64, i64) -> i64,
    {
        match (self.lhs, self.rhs) {
            (Some(lhs_time), Some(rhs_time)) => Time {
                time: Some(f(lhs_time, rhs_time)),
                time_base: self.time_base,
            },
            _ => Time {
                time: None,
                time_base: self.time_base,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let time = Time::new(Some(2), new_rational(3, 9));
        assert!(time.has_value());
        assert_eq!(time.as_secs(), 2.0 / 3.0);
        assert_eq!(time.into_value(), Some(2));
    }

    #[test]
    fn test_aligned_with_rational() {
        let time = Time::new(Some(2), new_rational(3, 9));
        assert_eq!(time.as_secs(), 2.0 / 3.0);
        let time = time.aligned_with_rational(new_rational(1, 9));
        assert_eq!(time.as_secs(), 2.0 / 3.0);
        assert_eq!(time.into_value(), Some(6));
    }

    #[test]
    fn test_from_nth_of_a_second() {
        let time = Time::from_nth_of_a_second(4);
        assert!(time.has_value());
        assert_eq!(time.as_secs(), 0.25);
        assert_eq!(time.as_secs_f64(), 0.25);
        assert_eq!(Duration::from(time), Duration::from_millis(250));
    }

    #[test]
    fn test_from_secs() {
        let time = Time::from_secs(2.5);
        assert!(time.has_value());
        assert_eq!(time.as_secs(), 2.5);
        assert_eq!(time.as_secs_f64(), 2.5);
        assert_eq!(Duration::from(time), Duration::from_millis(2500));
    }

    #[test]
    fn test_from_secs_f64() {
        let time = Time::from_secs(4.0);
        assert!(time.has_value());
        assert_eq!(time.as_secs_f64(), 4.0);
    }

    #[test]
    fn test_from_units() {
        let time = Time::from_units(3, 5);
        assert!(time.has_value());
        assert_eq!(time.as_secs(), 3.0 / 5.0);
        assert_eq!(Duration::from(time), Duration::from_millis(600));
    }

    #[test]
    fn test_zero() {
        let time = Time::zero();
        assert!(time.has_value());
        assert_eq!(time.as_secs(), 0.0);
        assert_eq!(time.as_secs_f64(), 0.0);
        assert_eq!(Duration::from(time), Duration::ZERO);
        let time = Time::zero();
        assert_eq!(time.into_value(), Some(0));
    }

    /// `Time::zero()` 与同样表示 0 秒的值相等（无论时间基是否相同）。
    #[test]
    fn test_zero_equals_other_zero_values() {
        assert_eq!(Time::zero(), Time::new(Some(0), TIME_BASE));
        assert_eq!(Time::zero(), Time::from_secs(0.0));
        assert_eq!(Time::zero(), Time::from_secs_f64(0.0));
    }

    /// "无值" 与 `AV_NOPTS_VALUE` 都表示"没有可用时间戳"：`new` 在 FFI 边界把
    /// 哨兵归一为 `None`，所有取值口（`has_value` / `into_value` / 秒换算）一致。
    #[test]
    fn test_has_value_rejects_missing_and_nopts() {
        let missing = Time::new(None, TIME_BASE);
        let nopts = Time::new(Some(ffi::AV_NOPTS_VALUE), TIME_BASE);

        assert!(!missing.has_value());
        assert!(!nopts.has_value());
        assert_eq!(nopts.into_value(), None);
        assert_eq!(nopts.as_secs_f64(), 0.0);
        assert_eq!(nopts.as_secs(), 0.0);
        assert_eq!(nopts.to_string(), "none");
        assert_eq!(nopts, missing);

        assert!(Time::new(Some(0), TIME_BASE).has_value());
    }

    /// `partial_cmp` 与 `PartialEq` 同键（秒）：同一时刻即使时间基不同也相等且有序。
    #[test]
    fn test_partial_cmp_is_consistent_with_eq() {
        let a = Time::new(None, new_rational(1, 2));
        let b = Time::new(None, new_rational(1, 4));
        assert_eq!(a, b, "两个无值的时间戳相等");
        assert_eq!(
            a.partial_cmp(&b),
            Some(std::cmp::Ordering::Equal),
            "无值之间可比且相等"
        );

        let same = Time::new(None, new_rational(1, 2));
        assert_eq!(a, same);
        assert_eq!(a.partial_cmp(&same), Some(std::cmp::Ordering::Equal));

        let later = Time::new(Some(3), new_rational(1, 2));
        assert_eq!(a.partial_cmp(&later), Some(std::cmp::Ordering::Less));
        assert_eq!(later.partial_cmp(&a), Some(std::cmp::Ordering::Greater));

        // 同一时刻、时间基不同 —— 秒是唯一比较键
        let same_instant = new_rational(1, 4);
        let later_other_base = Time::new(Some(6), same_instant);
        assert_eq!(later, later_other_base);
        assert_eq!(
            later.partial_cmp(&later_other_base),
            Some(std::cmp::Ordering::Equal)
        );

        let earlier = Time::new(Some(1), new_rational(1, 4));
        assert_eq!(
            earlier.partial_cmp(&later_other_base),
            Some(std::cmp::Ordering::Less)
        );
        assert!(earlier < later_other_base);
    }

    /// `Display` 打印秒数且不会因 `time * time_base.num` 溢出 `i64` 而 panic。
    #[test]
    fn test_display_prints_seconds_without_overflow() {
        assert_eq!(Time::new(Some(2), new_rational(1, 2)).to_string(), "1 secs");
        assert_eq!(Time::new(None, TIME_BASE).to_string(), "none");
        // 旧实现会计算 `time_base.num as i64 * time`，在 debug 下 panic
        let huge = Time::new(Some(i64::MAX / 2), new_rational(1_000_000, 1_000_000));
        assert!(huge.to_string().ends_with(" secs"));
    }

    #[test]
    fn test_aligned_with() {
        let a = Time::from_units(3, 16);
        let b = Time::from_units(1, 8);
        let aligned = a.aligned_with(b);
        assert_eq!(aligned.lhs, Some(3));
        assert_eq!(aligned.rhs, Some(2));
    }

    #[test]
    fn test_into_aligned_with() {
        let a = Time::from_units(2, 7);
        let b = Time::from_units(2, 3);
        let aligned = a.aligned_with(b);
        assert_eq!(aligned.lhs, Some(2));
        assert_eq!(aligned.rhs, Some(5));
    }

    #[test]
    fn test_as_secs() {
        let time = Time::from_nth_of_a_second(4);
        assert_eq!(time.as_secs(), 0.25);
        let time = Time::from_secs(0.3);
        assert_eq!(time.as_secs(), 0.3);
        let time = Time::new(None, new_rational(0, 0));
        assert_eq!(time.as_secs(), 0.0);
    }

    #[test]
    fn test_as_secs_f64() {
        let time = Time::from_nth_of_a_second(4);
        assert_eq!(time.as_secs_f64(), 0.25);
        let time = Time::from_secs_f64(0.3);
        assert_eq!(time.as_secs_f64(), 0.3);
        let time = Time::new(None, new_rational(0, 0));
        assert_eq!(time.as_secs_f64(), 0.0);
    }

    #[test]
    fn test_into_value_none() {
        let time = Time::new(None, new_rational(0, 0));
        assert_eq!(time.into_value(), None);
    }

    #[test]
    fn test_add() {
        let a = Time::from_secs(0.2);
        let b = Time::from_secs(0.3);
        assert_eq!(a.aligned_with(b).add(), Time::from_secs(0.5));
    }

    #[test]
    fn test_subtract() {
        let a = Time::from_secs(0.8);
        let b = Time::from_secs(0.4);
        assert_eq!(a.aligned_with(b).subtract(), Time::from_secs(0.4));
    }

    #[test]
    fn test_apply() {
        let a = Time::from_secs(2.0);
        let b = Time::from_secs(0.25);
        assert_eq!(
            a.aligned_with(b).apply(|x, y| (2 * x) + (3 * y)),
            Time::from_secs(4.75)
        );
    }

    #[test]
    fn test_apply_different_time_bases() {
        let a = Time::new(Some(3), new_rational(2, 32));
        let b = Time::from_nth_of_a_second(4);
        assert!(
            (a.aligned_with(b).apply(|x, y| x + y).as_secs()
                - Time::from_secs(7.0 / 16.0).as_secs())
            .abs()
                < 0.001
        );
    }

    #[test]
    fn test_negative_into_duration_clamps() {
        assert_eq!(
            Duration::from(Time::new(Some(-100), new_rational(0, 0))),
            Duration::ZERO,
        )
    }

    /// `AV_NOPTS_VALUE` 是"无时间戳"，在 FFI 边界归一为 `None`：所有取值口都
    /// 按"无值"处理（早先 `into_value` 会把哨兵原样吐出来，与 `has_value` 矛盾）。
    #[test]
    fn test_av_no_pts_value_is_normalized() {
        let nopts = Time::new(Some(ffi::AV_NOPTS_VALUE), new_rational(0, 0));
        assert_eq!(nopts.time, None);
        assert!(!nopts.has_value());
        assert_eq!(nopts.into_value(), None);
        assert_eq!(nopts.as_secs_f64(), 0.0);
        assert_eq!(Duration::from(nopts).as_secs_f32(), 0.0);

        // 公开字段仍可手工塞入哨兵，取值口同样归一（不 panic、不吐出约 -9.2e13 秒）
        let hand_built = Time {
            time: Some(ffi::AV_NOPTS_VALUE),
            time_base: TIME_BASE,
        };
        assert!(!hand_built.has_value());
        assert_eq!(hand_built.into_value(), None);
        assert_eq!(hand_built.as_secs_f64(), 0.0);
    }
}
