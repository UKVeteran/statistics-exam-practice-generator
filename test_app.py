import io
import matplotlib.pyplot as plt
import numpy as np
import pytest

from app import (
    create_teacher_pdf,
    generate_question,
    plot_boxplot,
    plot_histogram,
    plot_normal_dist,
)


@pytest.mark.parametrize("difficulty", ["Easy", "Intermediate", "Advanced"])
def test_generate_question_structure(difficulty):
    q = generate_question(difficulty)

    assert "data" in q
    assert "type" in q
    assert "mean" in q
    assert "median" in q
    assert "q1" in q
    assert "q3" in q
    assert "iqr" in q
    assert "std" in q

    assert isinstance(q["data"], np.ndarray)
    assert len(q["data"]) > 0
    assert isinstance(q["mean"], float)
    assert isinstance(q["iqr"], float)


def test_statistical_calculations_accuracy():
    q = generate_question("Easy")
    data = q["data"]

    expected_iqr = q["q3"] - q["q1"]
    assert pytest.approx(q["iqr"], rel=1e-5) == expected_iqr

    if len(data) > 1:
        expected_std = float(np.std(data, ddof=1))
        assert pytest.approx(q["std"], rel=1e-5) == expected_std


def test_plot_generators():
    data = np.array([10, 20, 30, 40, 50])

    fig_box = plot_boxplot(data)
    assert isinstance(fig_box, plt.Figure)
    plt.close(fig_box)

    fig_hist = plot_histogram(data)
    assert isinstance(fig_hist, plt.Figure)
    plt.close(fig_hist)

    fig_norm = plot_normal_dist(mu=50, sigma=10, cutoff=45)
    assert isinstance(fig_norm, plt.Figure)
    plt.close(fig_norm)


def test_teacher_pdf_generation():
    q = generate_question("Intermediate")
    pdf_buffer = create_teacher_pdf(q)

    assert isinstance(pdf_buffer, io.BytesIO)
    assert pdf_buffer.getbuffer().nbytes > 0
