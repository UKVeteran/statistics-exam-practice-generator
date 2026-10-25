import io
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# --- Page Config ---
st.set_page_config(
    page_title="Statistics Exam Practice Generator",
    page_icon="📊",
    layout="wide",
)


# --- Plot Generators ---
def plot_boxplot(data):
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.boxplot(data, vert=False, patch_artist=True, boxprops=dict(facecolor="skyblue"))
    ax.set_title("Dataset Box Plot")
    ax.set_xlabel("Values")
    plt.tight_layout()
    return fig


def plot_histogram(data):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(data, bins=5, edgecolor="black", color="orange")
    ax.set_title("Dataset Histogram")
    ax.set_xlabel("Intervals")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    return fig


def plot_normal_dist(mu, sigma, cutoff):
    fig, ax = plt.subplots(figsize=(6, 3))
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
    y = stats.norm.pdf(x, mu, sigma)
    ax.plot(x, y, color="black", lw=2)
    
    # Shade area below cutoff
    x_fill = np.linspace(mu - 4 * sigma, cutoff, 500)
    y_fill = stats.norm.pdf(x_fill, mu, sigma)
    ax.fill_between(x_fill, y_fill, color="skyblue", alpha=0.5, label=f"P(X ≤ {cutoff})")
    
    ax.set_title(f"Normal Distribution: N({mu}, {sigma}²)")
    ax.axvline(cutoff, color="red", linestyle="--", label=f"X = {cutoff}")
    ax.legend(loc="upper right")
    plt.tight_layout()
    return fig


# --- Question Generator Logic ---
def generate_question(difficulty):
    np.random.seed(random.randint(0, 10000))

    if difficulty == "Easy":
        data = np.random.randint(10, 50, size=10)
        q_type = random.choice(["summary", "normal_prob"])
    elif difficulty == "Intermediate":
        data = np.random.randint(20, 100, size=15)
        q_type = random.choice(["summary", "boxplot", "normal_prob", "ttest"])
    else:  # Advanced
        data = np.random.exponential(scale=25, size=20).astype(int) + 10
        q_type = random.choice(["summary", "boxplot", "histogram", "normal_prob", "ttest"])

    data = np.sort(data)

    # Calculate standard metrics
    mean_val = float(np.mean(data))
    median_val = float(np.median(data))
    q1 = float(np.percentile(data, 25))
    q3 = float(np.percentile(data, 75))
    iqr_val = float(q3 - q1)
    std_val = float(np.std(data, ddof=1)) if len(data) > 1 else 1.0

    # Normal distribution parameters
    norm_mu = random.randint(50, 100)
    norm_sigma = random.randint(5, 15)
    norm_cutoff = norm_mu + random.choice([-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]) * norm_sigma
    z_score = float((norm_cutoff - norm_mu) / norm_sigma)
    norm_prob = float(stats.norm.cdf(z_score))

    # T-test parameters
    null_mean = float(np.round(mean_val + random.choice([-3, -2, 2, 3]), 2))
    t_stat, p_val = stats.ttest_1samp(data, null_mean)

    return {
        "data": data,
        "type": q_type,
        "mean": mean_val,
        "median": median_val,
        "q1": q1,
        "q3": q3,
        "iqr": iqr_val,
        "std": std_val,
        "norm_mu": norm_mu,
        "norm_sigma": norm_sigma,
        "norm_cutoff": norm_cutoff,
        "z_score": z_score,
        "norm_prob": norm_prob,
        "null_mean": null_mean,
        "t_stat": float(t_stat),
        "p_val": float(p_val),
    }


# --- ReportLab PDF Generator ---
def create_teacher_pdf(q):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, height - 50, "Statistics Practice Worksheet & Answer Key")

    p.setFont("Helvetica-Bold", 12)
    p.drawString(50, height - 85, "Section A: Descriptive Statistics & Hypothesis Testing")
    p.setFont("Helvetica", 10)
    p.drawString(50, height - 105, f"Dataset: {', '.join(map(str, q['data'].tolist()))}")
    p.drawString(50, height - 125, "1. Calculate the sample mean and standard deviation.")
    p.drawString(50, height - 145, f"2. Perform a 1-sample t-test testing H0: mu = {q['null_mean']}. Find the t-statistic.")

    p.setFont("Helvetica-Bold", 12)
    p.drawString(50, height - 180, "Section B: Normal Distribution Probabilities")
    p.setFont("Helvetica", 10)
    p.drawString(50, height - 200, f"Assume X ~ N({q['norm_mu']}, {q['norm_sigma']}^2).")
    p.drawString(50, height - 220, f"3. Calculate the z-score and probability P(X <= {q['norm_cutoff']:.1f}).")

    # Answer Key Section
    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, height - 270, "--- Teacher Answer Key ---")
    p.setFont("Helvetica", 10)
    p.drawString(50, height - 295, f"Sample Mean: {q['mean']:.2f} | Sample Std Dev: {q['std']:.2f}")
    p.drawString(50, height - 315, f"1-Sample t-statistic (H0: mu = {q['null_mean']}): t = {q['t_stat']:.3f} (p-val = {q['p_val']:.4f})")
    p.drawString(50, height - 335, f"Z-score: {q['z_score']:.2f}")
    p.drawString(50, height - 355, f"Probability P(X <= {q['norm_cutoff']:.1f}): {q['norm_prob']:.4f}")

    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer


# --- Streamlit UI App ---
st.title("📊 Statistics Exam Practice Generator")
st.markdown(
    "Practice descriptive statistics, probability distributions, visual data plots, and hypothesis testing."
)

# Sidebar Controls
st.sidebar.header("Configuration")
mode = st.sidebar.radio("Select Mode", ["Student Practice", "Teacher PDF Hub"])
difficulty = st.sidebar.selectbox(
    "Difficulty Level", ["Easy", "Intermediate", "Advanced"]
)

if "q" not in st.session_state or st.sidebar.button("Generate New Question"):
    st.session_state.q = generate_question(difficulty)

q = st.session_state.q

if mode == "Student Practice":
    st.subheader(f"Practice Task ({difficulty} Level)")

    if q["type"] in ["summary", "boxplot", "histogram"]:
        st.write("Given the following dataset, solve the question below:")
        st.code(list(q["data"]))

        if q["type"] == "boxplot":
            st.pyplot(plot_boxplot(q["data"]))
        elif q["type"] == "histogram":
            st.pyplot(plot_histogram(q["data"]))

        col1, col2 = st.columns(2)
        with col1:
            u_mean = st.number_input("Enter the Sample Mean:", format="%.2f")
        with col2:
            u_iqr = st.number_input("Enter the Interquartile Range (IQR):", format="%.2f")

        if st.button("Submit Answer"):
            if abs(u_mean - q["mean"]) <= 0.1 and abs(u_iqr - q["iqr"]) <= 0.1:
                st.success("🎉 Correct!")
            else:
                st.error("❌ Incorrect. Check the worked solution below.")

        with st.expander("💡 Need a Hint?"):
            st.write(f"Sum = `{np.sum(q['data'])}`, Count ($n$) = `{len(q['data'])}`.")
            st.write(f"Q1 = `{q['q1']}`, Q3 = `{q['q3']}`.")

        with st.expander("📖 View Worked Solution"):
            st.write(f"- **Mean:** $\\bar{{x}} = \\frac{{\\sum x}}{{n}} = {q['mean']:.2f}$")
            st.write(f"- **IQR:** $Q3 - Q1 = {q['q3']} - {q['q1']} = {q['iqr']:.2f}$")

    elif q["type"] == "normal_prob":
        st.markdown(
            f"A continuous random variable $X$ is normally distributed with mean **$\\mu = {q['norm_mu']}$** and standard deviation **$\\sigma = {q['norm_sigma']}$**."
        )
        st.pyplot(plot_normal_dist(q["norm_mu"], q["norm_sigma"], q["norm_cutoff"]))

        col1, col2 = st.columns(2)
        with col1:
            u_z = st.number_input(f"1. Calculate the Z-score for X = {q['norm_cutoff']:.1f}:", format="%.2f")
        with col2:
            u_p = st.number_input(f"2. Calculate P(X ≤ {q['norm_cutoff']:.1f}):", format="%.4f")

        if st.button("Submit Answer"):
            if abs(u_z - q["z_score"]) <= 0.05 and abs(u_p - q["norm_prob"]) <= 0.02:
                st.success("🎉 Correct!")
            else:
                st.error("❌ Incorrect. Check the worked solution below.")

        with st.expander("💡 Need a Hint?"):
            st.write(r"Formula for Z-score: $Z = \frac{X - \mu}{\sigma}$")

        with st.expander("📖 View Worked Solution"):
            st.write(f"- **Z-score:** $Z = \\frac{{{q['norm_cutoff']:.1f} - {q['norm_mu']}}}{{{q['norm_sigma']}}} = {q['z_score']:.2f}$")
            st.write(f"- **Cumulative Probability:** $P(Z \\le {q['z_score']:.2f}) = {q['norm_prob']:.4f}$")

    elif q["type"] == "ttest":
        st.write("Given the sample dataset:")
        st.code(list(q["data"]))
        st.markdown(
            f"Test the null hypothesis **$H_0: \\\\mu = {q['null_mean']}$** "
            f"against the two-sided alternative **$H_1: \\\\mu \\\\neq {q['null_mean']}$**."
        )

        col1, col2 = st.columns(2)
        with col1:
            u_t = st.number_input("Enter the calculated t-statistic:", format="%.3f")
        with col2:
            u_p = st.number_input("Enter the calculated p-value:", format="%.4f")

        if st.button("Submit Answer"):
            if abs(u_t - q["t_stat"]) <= 0.05 and abs(u_p - q["p_val"]) <= 0.02:
                st.success("🎉 Correct!")
            else:
                st.error("❌ Incorrect. Check the worked solution below.")

        with st.expander("💡 Need a Hint?"):
            st.write(r"Formula for 1-sample t-test statistic: $t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}$")
            st.write(f"Sample mean $\\bar{{x}} = {q['mean']:.2f}$, Sample std dev $s = {q['std']:.2f}$, $n = {len(q['data'])}$.")

        with st.expander("📖 View Worked Solution"):
            st.write(
                f"- **t-statistic calculation:** $t = \\frac{{{q['mean']:.2f} - {q['null_mean']}}}{{{q['std']:.2f} / \\sqrt{{{len(q['data'])}}}}} = {q['t_stat']:.3f}$"
            )
            st.write(f"- **p-value:** $p = {q['p_val']:.4f}$ (Degrees of freedom $df = {len(q['data']) - 1}$)")

else:
    st.subheader("Teacher Worksheet & Answer Key Hub")
    st.markdown("Generate printable PDF worksheets with embedded answer keys for your class.")

    pdf_buffer = create_teacher_pdf(q)
    st.download_button(
        label="📥 Download Printable Worksheet & Answer Key (PDF)",
        data=pdf_buffer,
        file_name="statistics_worksheet.pdf",
        mime="application/pdf",
    )
