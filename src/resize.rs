/// Represents width and height in a tuple.
type Dims = (u32, u32);

/// Represents the possible resize strategies.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Resize {
    /// When resizing with `Resize::Exact`, each frame will be resized to the exact width and height
    /// given, without taking into account aspect ratio.
    Exact(u32, u32),
    /// When resizing with `Resize::Fit`, each frame will be resized to the biggest width and height
    /// within the given dimensions that keeps the aspect ratio **without upscaling**: a frame that
    /// already fits is returned unchanged, so the result is never larger than the source.
    Fit(u32, u32),
    /// Like [`Resize::Fit`], but both sides are then rounded **down** to an even value
    /// independently, so the aspect ratio holds only up to one pixel. Resizing using
    /// this method can fail if there exist no dimensions that fit these constraints.
    ///
    /// Note that this resizing method is especially useful since some encoders only accept frames
    /// with dimensions that are divisible by 2.
    FitEven(u32, u32),
}

impl Resize {
    /// Compute the dimensions after resizing depending on the resize strategy.
    ///
    /// # Arguments
    ///
    /// * `dims` - Input dimensions (width and height).
    ///
    /// # Return value
    ///
    /// Tuple of width and height with dimensions after resizing.
    pub fn compute_for(self, dims: Dims) -> Option<Dims> {
        match self {
            Resize::Exact(w, h) => Some((w, h)),
            Resize::Fit(w, h) => calculate_fit_dims(dims, (w, h)),
            Resize::FitEven(w, h) => calculate_fit_dims_even(dims, (w, h)),
        }
    }
}

/// Calculates the image dimensions `w` and `h` that fit inside `w_max` and `h_max`, retaining the
/// original aspect ratio and **never upscaling**: a source that already fits comes back verbatim,
/// so the result is bounded by the source as well as by `w_max` / `h_max`.
///
/// # Arguments
///
/// * `dims` - Original dimensions: width and height.
/// * `fit_dims` - Dimensions to fit in: width and height.
///
/// # Return value
///
/// The fitted dimensions if they exist and are positive and more than zero.
fn calculate_fit_dims(dims: (u32, u32), fit_dims: (u32, u32)) -> Option<(u32, u32)> {
    let (w, h) = dims;
    let (w_max, h_max) = fit_dims;
    if w == 0 || h == 0 {
        return None;
    }
    if w_max >= w && h_max >= h {
        Some((w, h))
    } else {
        let wf = w_max as f32 / w as f32;
        let hf = h_max as f32 / h as f32;
        let f = wf.min(hf);
        let (w_out, h_out) = ((w as f32 * f) as u32, (h as f32 * f) as u32);
        if (w_out > 0) && (h_out > 0) {
            Some((w_out, h_out))
        } else {
            None
        }
    }
}

/// Calculates the image dimensions `w` and `h` that fit inside `w_max` and `h_max` (never
/// upscaling, see [`calculate_fit_dims`]), where both the width and height must be divisble by two.
///
/// The two sides are rounded down to an even value **independently** after the single scaling pass,
/// so the aspect ratio is only preserved up to one pixel (`100x99` in `100x100` gives `100x98`).
///
/// Note that this method will even reduce the dimensions to even width and height if they already
/// fit in `fit_dims`.
///
/// # Arguments
///
/// * `dims` - Original dimensions: width and height.
/// * `fit_dims` - Dimensions to fit in: width and height.
///
/// # Return value
///
/// The fitted dimensions if they exist and are positive and more than zero.
fn calculate_fit_dims_even(dims: (u32, u32), fit_dims: (u32, u32)) -> Option<(u32, u32)> {
    let (w, h) = dims;
    if w == 0 || h == 0 {
        return None;
    }

    // 只缩放一次（与 `calculate_fit_dims` 相同，且绝不放大），然后把宽、高**各自**
    // 向下取偶。逐格压低 `w_max`/`h_max` 去凑偶数的旧做法会一路缩过头：
    // 100x99 配 100x100 会一直缩到 50x50，而不是 100x98。
    let (out_w, out_h) = calculate_fit_dims(dims, fit_dims)?;
    let (out_w, out_h) = (out_w & !1, out_h & !1);
    if out_w == 0 || out_h == 0 {
        None
    } else {
        Some((out_w, out_h))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TESTING_DIM_CANDIDATES: [u32; 8] = [0, 1, 2, 3, 8, 111, 256, 1000];

    #[test]
    fn calculate_fit_dims_works() {
        let testset = generate_testset();
        for ((w, h), (fit_w, fit_h)) in testset {
            let out = calculate_fit_dims((w, h), (fit_w, fit_h));
            if let Some((out_w, out_h)) = out {
                let input_dim_zero = w == 0 || h == 0 || fit_w == 0 || fit_h == 0;
                let output_dim_zero = out_w == 0 || out_h == 0;
                assert!(
                    (input_dim_zero && output_dim_zero) || (!input_dim_zero && !output_dim_zero),
                    "computed dims are never zero unless the inputs dims were",
                );
                assert!(
                    (out_w <= fit_w) && (out_h <= fit_h),
                    "computed dims fit inside provided dims",
                );
            }
        }
    }

    #[test]
    fn calculate_fit_dims_even_works() {
        let testset = generate_testset();
        for ((w, h), (fit_w, fit_h)) in testset {
            let out = calculate_fit_dims_even((w, h), (fit_w, fit_h));
            if let Some((out_w, out_h)) = out {
                let input_dim_zero = w == 0 || h == 0 || fit_w == 0 || fit_h == 0;
                let output_dim_zero = out_w == 0 || out_h == 0;
                assert!(
                    (input_dim_zero && output_dim_zero) || (!input_dim_zero && !output_dim_zero),
                    "computed dims are never zero unless the inputs dims were",
                );
                assert!(
                    (out_w % 2 == 0) && (out_h % 2 == 0),
                    "computed dims are even",
                );
                assert!(
                    (out_w <= fit_w) && (out_h <= fit_h),
                    "computed dims fit inside provided dims",
                );
            }
        }
    }

    /// 契约：尺寸算不出来就必须是 `None`，一个 0 宽或 0 高的结果**不能**从
    /// `Some` 里出来 —— `Decoder` 正是靠 `None` 把"这套 resize 配置算不出可用
    /// 尺寸"报成 `InvalidConfig`（`decode.rs` 的 `compute_for`），`Some((0, 0))`
    /// 会一路走到 scaler 去。
    #[test]
    fn degenerate_source_dims_are_none_not_zero() {
        for (dims, fit) in [
            ((0, 0), (1000, 1000)),
            ((0, 480), (1000, 1000)),
            ((640, 0), (1000, 1000)),
            ((0, 0), (0, 0)),
        ] {
            assert_eq!(
                calculate_fit_dims(dims, fit),
                None,
                "{dims:?} in {fit:?} has no positive fitted size"
            );
            assert_eq!(
                calculate_fit_dims_even(dims, fit),
                None,
                "{dims:?} in {fit:?} has no positive even fitted size"
            );
            assert_eq!(
                Resize::Fit(fit.0, fit.1).compute_for(dims),
                None,
                "Resize::Fit{fill:?} on {dims:?}",
                fill = fit
            );
        }
    }

    /// `Fit` 绝不放大：源本身就装得下时原样返回（文档里的"biggest ... within the
    /// given dimensions"只在这个前提下成立）。
    #[test]
    fn fit_never_upscales() {
        assert_eq!(
            Resize::Fit(1000, 1000).compute_for((640, 480)),
            Some((640, 480)),
            "a source that already fits is returned unchanged, not 1000x750"
        );
        assert_eq!(
            Resize::Fit(320, 240).compute_for((640, 480)),
            Some((320, 240))
        );
    }

    fn generate_testset() -> Vec<((u32, u32), (u32, u32))> {
        let testing_dims = generate_testing_dims();
        testing_dims
            .iter()
            .flat_map(|dims| testing_dims.iter().map(|fit_dims| (*dims, *fit_dims)))
            .collect()
    }

    fn generate_testing_dims() -> Vec<(u32, u32)> {
        TESTING_DIM_CANDIDATES
            .iter()
            .flat_map(|a| TESTING_DIM_CANDIDATES.iter().map(|b| (*a, *b)))
            .collect()
    }
}
