/*
** +---------------------------------------------------------------------+
** | (c) 2025 Mario Sieg <mario.sieg.64@gmail.com>                       |
** | Licensed under the Apache License, Version 2.0                      |
** |                                                                     |
** | Website : https://mariosieg.com                                     |
** | GitHub  : https://github.com/MarioSieg                              |
** | License : https://www.apache.org/licenses/LICENSE-2.0               |
** +---------------------------------------------------------------------+
*/

#include <prelude.hpp>

// This file (because of Boost) increases the compile time diabolically,
// so just enable these tests if you messed with the prng
#if 0

#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/uniform.hpp>
#include <boost/math/distributions/binomial.hpp>
#include <boost/math/distributions/kolmogorov_smirnov.hpp>

using namespace magnetron;

static constexpr size_t k_iter_samples = 10;

[[nodiscard]] static auto bernoulli_two_sided_pvalue(uint64_t N, uint64_t k, double p) -> double {
    boost::math::binomial_distribution<> binom{static_cast<double>(N), p};
    double cdf_lo = boost::math::cdf(binom, static_cast<double>(k));
    double cdf_hi = 1.0 - boost::math::cdf(binom, static_cast<double>(k - 1));
    double p_two = 2.0*std::min(cdf_lo, cdf_hi);
    return std::min(1.0, p_two);
}

template <typename T, typename Cdf>
[[nodiscard]] static auto ks_test_p_value(const std::vector<T>& samples, const Cdf& cdf_theory) -> double {
    std::vector<double> x(samples.begin(), samples.end());
    std::sort(x.begin(), x.end());
    double n = static_cast<double>(x.size());
    double D = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        double F  = static_cast<double>(cdf_theory(x[i]));
        double Fn_above = static_cast<double>(i + 1) / n;
        double Fn_below = static_cast<double>(i) / n;
        double d1 = std::fabs(Fn_above - F);
        double d2 = std::fabs(F - Fn_below);
        D = std::max(D, std::max(d1, d2));
    }
    boost::math::kolmogorov_smirnov_distribution<> ks(x.size());
    return 1.0 - cdf(ks, D);
}

[[nodiscard]] static auto allowed_failures_3sigma(size_t trials, double alpha) -> int32_t {
    double mu = static_cast<double>(trials)*alpha;
    double var = static_cast<double>(trials)*alpha * (1.0 - alpha);
    double ub = mu + 3.0*std::sqrt(std::max(0.0, var));
    return static_cast<int32_t>(std::ceil(ub));
}

TEST(prng, ks_test_normal_dist) {
    std::mt19937_64 eng{0x9e3779b97f4a7c15ULL};
    std::uniform_real_distribution<double> mean_d{-3.0, 3.0};
    std::uniform_real_distribution<double> std_d{0.1, 3.0};
    std::uniform_int_distribution<int64_t> shape_distr{512, 1024};
    context ctx{};
    ctx.manual_seed(0x1234567890abcdefULL);
    constexpr double alpha = 0.01;
    int32_t failures = 0;
    struct Hit { double p; double mu, sigma; size_t rows, cols; };
    std::vector<Hit> worst;
    for (size_t i = 0; i < k_iter_samples; ++i) {
        double mean = mean_d(eng);
        double stdv = std_d(eng);
        size_t rows = shape_distr(eng);
        size_t cols = shape_distr(eng);
        tensor t{ctx, dtype::float32, rows, cols};
        t.normal_(static_cast<float>(mean), static_cast<float>(stdv));
        std::vector<float> samples = t.to_vector<float>();
        const boost::math::normal dist(mean, stdv);
        auto cdf = [&](double x) { return boost::math::cdf(dist, x); };
        const double p_value = ks_test_p_value(samples, cdf);
        worst.push_back({p_value, mean, stdv, rows, cols});
        if (p_value <= alpha) ++failures;
    }
    std::ranges::sort(worst, [](const Hit& a, const Hit& b){ return a.p < b.p; });
    std::ostringstream msg;
    const size_t show = std::min<size_t>(5, worst.size());
    msg << "Worst p-values (alpha=" << alpha << "):\n";
    for (size_t i = 0; i < show; ++i) {
        msg << "  p=" << worst[i].p
            << "  mu=" << worst[i].mu
            << "  sigma=" << worst[i].sigma
            << "  shape=" << worst[i].rows << "x" << worst[i].cols << "\n";
    }
    const int32_t allowed = allowed_failures_3sigma(k_iter_samples, alpha);
    EXPECT_LE(failures, allowed) << "KS: too many low p-values across sweeps.\n"
                                 << "failures=" << failures
                                 << " allowed≤" << allowed << "\n"
                                 << msg.str();
}

TEST(prng, ks_test_uniform_dist) {
    std::mt19937_64 eng{0x9e3779b97f4a7c15ULL};
    std::uniform_real_distribution<double> a_d{-5.0, 0.0};
    std::uniform_real_distribution<double> w_d{0.1, 5.0};
    std::uniform_int_distribution<int64_t> shape_d{512, 1024};
    context ctx{};
    ctx.manual_seed(0x1234567890abcdefULL);
    int32_t failures = 0;
    for (size_t it = 0; it < k_iter_samples; ++it) {
        double a = a_d(eng);
        double b = a + w_d(eng);
        size_t rows = shape_d(eng);
        size_t cols = shape_d(eng);
        tensor t{ctx, dtype::float32, rows, cols};
        t.uniform_(static_cast<float>(a), static_cast<float>(b));
        std::vector<float> samples = t.to_vector<float>();
        boost::math::uniform_distribution<> dist(a, b);
        auto cdf = [&](double x) { return boost::math::cdf(dist, x); };
        double p = ks_test_p_value(samples, cdf);
        if (p <= 0.001) ++failures;
    }
    EXPECT_LE(failures, 1) << "Too many KS failures for uniform sweeps";
}

TEST(prng, bernoulli_binomial_exact) {
    std::mt19937_64 eng{0x9e3779b97f4a7c15ULL};
    std::uniform_real_distribution<double> p_d{0.02, 0.98};
    std::uniform_int_distribution<int64_t> shape_d{512, 1024};
    context ctx{};
    ctx.manual_seed(0x1234567890abcdefULL);
    size_t failures = 0;
    for (size_t it = 0; it < k_iter_samples; ++it) {
        double p = p_d(eng);
        size_t rows = shape_d(eng);
        size_t cols = shape_d(eng);
        uint64_t N = static_cast<uint64_t>(rows) * cols;
        tensor t{ctx, dtype::boolean, rows, cols};
        t.bernoulli_(static_cast<float>(p));
        std::vector<bool> v = t.to_vector<bool>();
        uint64_t ones = 0;
        for (bool x : v) ones += x ? 1 : 0;
        double pv = bernoulli_two_sided_pvalue(N, ones, p);
        if (pv <= 0.001) ++failures;
    }
    EXPECT_LE(failures, 1) << "Too many Bernoulli (binomial) failures";
}

TEST(prng, normal_mean_std_match) {
    context ctx{};
    ctx.manual_seed(0x1234567890abcdefULL);
    constexpr size_t N = 1'000'000;
    constexpr double mu_true  = 1.5;
    constexpr double sigma_true = 2.0;
    tensor t{ctx, dtype::float32, N};
    t.normal_(mu_true, sigma_true);
    std::vector<float> v = t.to_vector<float>();
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mu_hat = sum / N;
    double sq_sum = 0.0;
    for (float x : v) {
        double d = x - mu_hat;
        sq_sum += d * d;
    }
    double sigma_hat = std::sqrt(sq_sum / (N - 1));
    double se_mean = sigma_true / std::sqrt(static_cast<double>(N));
    double se_std  = sigma_true * std::sqrt(1.0 / (2.0 * (N - 1)));
    EXPECT_NEAR(mu_hat, mu_true, 5.0 * se_mean) << "Mean deviates too much from target";
    EXPECT_NEAR(sigma_hat, sigma_true, 5.0 * se_std) << "Std dev deviates too much from target";
}

TEST(prng, automatic_seeding) {
    std::vector<float> a, b;
    {
        context ctx {};
        tensor ta {ctx, dtype::float32, 8192, 8192};
        ta.uniform_(-1.0f, 1.0f);
        a = ta.to_vector<float>();
    }
    {
        context ctx {};
        tensor tb {ctx, dtype::float32, 8192, 8192};
        tb.uniform_(-1.0f, 1.0f);
        b = tb.to_vector<float>();
    }

    ASSERT_EQ(a.size(), b.size());

    ASSERT_NE(0, std::memcmp(a.data(), b.data(), a.size() * sizeof(float)));

    size_t matches = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] == b[i]) ++matches;
    }
    EXPECT_LE(matches, 30u) << "Too many exact matches - suspicious seeding";
}

TEST(prng, manual_seeding) {
    std::random_device rd;
    std::mt19937_64 eng {rd()};
    std::uniform_int_distribution<uint64_t> distr {};
    uint64_t seed = distr(eng);
    std::vector<float> a {}, b {};
    {
        context ctx {};
        ctx.manual_seed(seed);
        tensor ta {ctx, dtype::float32, 8192, 8192};
        ta.uniform_(-1.0f, 1.0f);
        a = ta.to_vector<float>();
    }

    {
        context ctx {};
        ctx.manual_seed(seed);
        tensor tb {ctx, dtype::float32, 8192, 8192};
        tb.uniform_(-1.0f, 1.0f);
        b = tb.to_vector<float>();
    }

    ASSERT_EQ(a.size(), b.size());

    for (size_t i=0; i < a.size(); i++)
        ASSERT_FLOAT_EQ(a[i], b[i]) << "i=" << i;
}

#endif
