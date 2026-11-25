"""
Interface Prediksi Kualitas Apel
================================
Membaca model hasil notebook Klasifikasi_Kualitas_Apel.ipynb
(file model_kualitas_apel.joblib), lalu memprediksi kualitas apel
dari tujuh karakteristiknya.

Cara menjalankan, dari folder yang sama dengan file ini:

    pip install streamlit
    streamlit run app.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

FOLDER = Path(__file__).parent
BERKAS_MODEL = FOLDER / "model_kualitas_apel.joblib"
BERKAS_DATA = FOLDER / "apple_quality_clean.csv"

# Nama fitur dalam bahasa Indonesia untuk ditampilkan di layar
NAMA_FITUR = {
    "Size": "Ukuran",
    "Weight": "Berat",
    "Sweetness": "Kemanisan",
    "Crunchiness": "Kerenyahan",
    "Juiciness": "Kandungan air",
    "Ripeness": "Kematangan",
    "Acidity": "Keasaman",
}

# Batas tampilan skala skor keputusan (khusus model tanpa peluang, misalnya SVM)
BATAS_SKOR = 4.0

st.set_page_config(page_title="Prediksi Kualitas Apel", page_icon="🍏", layout="wide")

st.markdown(
    """
    <style>
      .block-container { max-width: 1080px; padding-top: 2.4rem; padding-bottom: 3rem; }
      h1, h3 { color: #1E5E24 !important; }
      h1 { letter-spacing: -0.02em; }

      /* Dua panel putih di atas latar hijau muda */
      .st-key-kartu_input, .st-key-kartu_hasil {
        background: #FFFFFF;
        border: 1px solid #D3E6C8 !important;
        border-radius: 18px !important;
        padding: 1.3rem 1.5rem 1.4rem !important;
      }

      /* Kotak hasil: satu-satunya bidang berwarna penuh di halaman */
      .hasil { border-radius: 14px; padding: 1.15rem 1.35rem; margin: .1rem 0 1.4rem 0; }
      .hasil, .hasil * { color: #FFFFFF !important; }
      .hasil-good { background: #2E7D32; }
      .hasil-bad  { background: #B23A32; }
      .hasil-judul { font-size: 2.05rem !important; font-weight: 800; line-height: 1.1; letter-spacing: -0.01em; }
      .hasil-sub { margin-top: .4rem; opacity: .92; }

      .skala { margin: .3rem 0 .4rem 0; }
      .skala-trek { position: relative; display: flex; height: 12px; }
      .skala-bagian { height: 12px; }
      .skala-bagian:first-child { border-radius: 6px 0 0 6px; }
      .skala-bagian.terakhir { border-radius: 0 6px 6px 0; }
      .merah-tua  { background: #B23A32; }
      .merah-muda { background: #EDC9C5; }
      .hijau-muda { background: #C7E3BF; }
      .hijau-tua  { background: #2E7D32; }
      .skala-tengah {
        position: absolute; left: 50%; top: -5px; width: 2px; height: 22px;
        margin-left: -1px; background: #1C3322; opacity: .55;
      }
      .skala-penanda {
        position: absolute; top: -7px; width: 16px; height: 26px; margin-left: -8px;
        box-sizing: border-box; border-radius: 8px; background: #FFFFFF;
        border: 3px solid #1C3322; box-shadow: 0 1px 4px rgba(28, 51, 34, .25);
      }
      .skala-label {
        display: flex; justify-content: space-between;
        font-size: .85rem; color: #5A6E5E; margin-top: .6rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Memuat model dan dataset
# ----------------------------------------------------------------------
@st.cache_resource
def muat_paket_model():
    return joblib.load(BERKAS_MODEL)


@st.cache_data
def muat_dataset():
    if BERKAS_DATA.exists():
        return pd.read_csv(BERKAS_DATA)
    return None


if not BERKAS_MODEL.exists():
    st.error(
        "File `model_kualitas_apel.joblib` belum ada di folder ini. "
        "Jalankan semua sel pada notebook `Klasifikasi_Kualitas_Apel.ipynb` "
        "sampai selesai, lalu muat ulang halaman ini."
    )
    st.stop()

try:
    paket = muat_paket_model()
except Exception as galat:  # biasanya karena versi scikit-learn berbeda
    st.error(
        "Model gagal dimuat. Jalankan ulang notebook di komputer ini supaya "
        "model dibuat dengan versi scikit-learn yang sama.\n\n"
        f"Rincian: `{galat}`"
    )
    st.stop()

model = paket["model"]
FITUR = paket["fitur"]
LABEL = paket["label"]          # contoh: {0: "bad", 1: "good"}
RENTANG = paket["rentang"]      # min, max, dan median setiap fitur
KODE_GOOD = next(kode for kode, nama in LABEL.items() if nama == "good")

dataset = muat_dataset()


# ----------------------------------------------------------------------
# Nilai awal dan tombol
# ----------------------------------------------------------------------
def kunci(fitur):
    return f"input_{fitur}"


def kunci_ketik(fitur):
    return f"ketik_{fitur}"


def rapikan(fitur, angka):
    """Membulatkan ke 2 desimal dan menjaga nilai tetap di dalam rentang dataset."""
    angka = round(float(angka), 2)
    return float(np.clip(angka, RENTANG[fitur]["min"], RENTANG[fitur]["max"]))


def atur_nilai(fitur, angka):
    """Slider dan kotak angka selalu diisi bersamaan supaya nilainya sama."""
    angka = rapikan(fitur, angka)
    st.session_state[kunci(fitur)] = angka
    st.session_state[kunci_ketik(fitur)] = angka
    return angka


def salin_dari_slider(fitur):
    atur_nilai(fitur, st.session_state[kunci(fitur)])


def salin_dari_ketik(fitur):
    atur_nilai(fitur, st.session_state[kunci_ketik(fitur)])


def kembalikan_nilai_tengah():
    for fitur in FITUR:
        atur_nilai(fitur, RENTANG[fitur]["median"])
    st.session_state["contoh"] = None


def ambil_contoh():
    baris = dataset.sample(1).iloc[0]
    nilai = {}
    for fitur in FITUR:
        nilai[fitur] = atur_nilai(fitur, baris[fitur])
    st.session_state["contoh"] = {
        "nilai": nilai,
        "label": str(baris["Quality"]),
        "nomor": int(baris["A_id"]),
    }


for fitur in FITUR:
    st.session_state.setdefault(kunci(fitur), RENTANG[fitur]["median"])
    st.session_state.setdefault(kunci_ketik(fitur), st.session_state[kunci(fitur)])
st.session_state.setdefault("contoh", None)


def persen(nilai):
    return f"{nilai * 100:.2f}%".replace(".", ",")


# ----------------------------------------------------------------------
# Tampilan
# ----------------------------------------------------------------------
st.title("Prediksi Kualitas Apel")
st.write(
    "Masukkan tujuh karakteristik apel dengan menggeser slider atau mengetik "
    "angkanya langsung di kotak angka. Hasil prediksi langsung berubah setiap "
    "kali ada nilai yang diubah."
)

kolom_input, kolom_hasil = st.columns([3, 2], gap="large")

with kolom_input, st.container(border=True, key="kartu_input"):
    st.subheader("Karakteristik apel")

    for fitur in FITUR:
        batas_bawah = float(RENTANG[fitur]["min"])
        batas_atas = float(RENTANG[fitur]["max"])
        kolom_geser, kolom_ketik = st.columns([3, 1], vertical_alignment="bottom")
        with kolom_geser:
            st.slider(
                f"{NAMA_FITUR.get(fitur, fitur)} ({fitur})",
                min_value=batas_bawah,
                max_value=batas_atas,
                step=0.01,
                key=kunci(fitur),
                on_change=salin_dari_slider,
                args=(fitur,),
            )
        with kolom_ketik:
            st.number_input(
                f"Ketik nilai {fitur}",
                min_value=batas_bawah,
                max_value=batas_atas,
                step=0.01,
                format="%.2f",
                key=kunci_ketik(fitur),
                on_change=salin_dari_ketik,
                args=(fitur,),
                label_visibility="collapsed",
            )

    tombol_kiri, tombol_kanan = st.columns(2)
    with tombol_kiri:
        if dataset is not None:
            st.button("Ambil contoh dari dataset", on_click=ambil_contoh, type="primary")
    with tombol_kanan:
        st.button("Kembalikan ke nilai tengah", on_click=kembalikan_nilai_tengah)

    st.caption(
        "Nilai pada dataset sudah berskala, bukan satuan seperti gram atau sentimeter, "
        "jadi angka negatif itu wajar. Batas slider dan kotak angka mengikuti nilai "
        "terkecil dan terbesar pada dataset, dan posisi awalnya adalah nilai tengah (median)."
    )

# Prediksi
nilai_sekarang = {fitur: float(st.session_state[kunci(fitur)]) for fitur in FITUR}
masukan = pd.DataFrame([[nilai_sekarang[fitur] for fitur in FITUR]], columns=FITUR)

kode_prediksi = int(model.predict(masukan)[0])
label_prediksi = LABEL[kode_prediksi]
prediksi_good = kode_prediksi == KODE_GOOD

with kolom_hasil, st.container(border=True, key="kartu_hasil"):
    st.subheader("Hasil prediksi")

    if prediksi_good:
        st.markdown(
            '<div class="hasil hasil-good"><div class="hasil-judul">Kualitas baik</div>'
            '<div class="hasil-sub">Model memprediksi apel ini <b>good</b>.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="hasil hasil-bad"><div class="hasil-judul">Kualitas buruk</div>'
            '<div class="hasil-sub">Model memprediksi apel ini <b>bad</b>.</div></div>',
            unsafe_allow_html=True,
        )

    # Seberapa yakin model? Peluang dipakai jika tersedia (Decision Tree, KNN).
    # SVM tidak menghasilkan peluang, jadi dipakai skor keputusannya.
    if hasattr(model, "predict_proba"):
        peluang = model.predict_proba(masukan)[0]
        peluang_good = float(peluang[list(model.classes_).index(KODE_GOOD)])
        posisi = peluang_good * 100
        segmen = [("merah-tua", 50), ("hijau-tua", 50)]
        keterangan = f"Peluang good menurut model: **{persen(peluang_good)}**"
        penjelasan = "Di atas 50% berarti good, di bawah 50% berarti bad."
        kurang_pasti = abs(peluang_good - 0.5) < 0.1
    elif hasattr(model, "decision_function"):
        skor = round(float(model.decision_function(masukan)[0]), 2)
        posisi = (float(np.clip(skor, -BATAS_SKOR, BATAS_SKOR)) + BATAS_SKOR) / (2 * BATAS_SKOR) * 100
        lebar_margin = 1 / (2 * BATAS_SKOR) * 100      # lebar bagian −1..0 dan 0..+1
        segmen = [
            ("merah-tua", 50 - lebar_margin), ("merah-muda", lebar_margin),
            ("hijau-muda", lebar_margin), ("hijau-tua", 50 - lebar_margin),
        ]
        keterangan = "Skor keputusan: **" + f"{skor:+.2f}".replace(".", ",") + "**"
        penjelasan = (
            "Skor positif berarti good dan skor negatif berarti bad. Makin jauh dari 0, "
            "makin yakin modelnya. Bagian berwarna pucat (−1 sampai +1) adalah margin SVM."
        )
        kurang_pasti = abs(skor) < 1
    else:
        posisi = None

    if posisi is not None:
        bagian = "".join(
            f'<div class="skala-bagian {warna}{" terakhir" if i == len(segmen) - 1 else ""}" '
            f'style="width:{lebar:.2f}%"></div>'
            for i, (warna, lebar) in enumerate(segmen)
        )
        st.markdown(
            f'<div class="skala"><div class="skala-trek">{bagian}'
            f'<div class="skala-tengah"></div>'
            f'<div class="skala-penanda" style="left:{posisi:.1f}%"></div></div>'
            f'<div class="skala-label"><span>bad</span><span>batas keputusan</span>'
            f'<span>good</span></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(keterangan)
        st.caption(penjelasan)
        if kurang_pasti:
            st.warning(
                "Apel ini dekat dengan batas keputusan, jadi prediksinya kurang pasti."
            )

    # Jika nilai berasal dari contoh dataset dan belum diubah, tampilkan label aslinya
    contoh = st.session_state["contoh"]
    if contoh is not None:
        belum_diubah = all(
            abs(nilai_sekarang[fitur] - contoh["nilai"][fitur]) < 0.005 for fitur in FITUR
        )
        if belum_diubah:
            sesuai = contoh["label"] == label_prediksi
            pesan = (
                f"Ini apel nomor {contoh['nomor']} pada dataset. "
                f"Label sebenarnya: **{contoh['label']}**. "
                + ("Prediksi model sesuai." if sesuai else "Prediksi model tidak sesuai.")
            )
            # Hijau kalau tebakan model benar, kuning kalau meleset
            (st.success if sesuai else st.warning)(pesan)

    with st.expander("Tentang model"):
        metrik = paket.get("metrik_uji", {})
        st.markdown(
            f"- Algoritma: **{paket.get('nama_model', '-')}**\n"
            f"- Data latih: {paket.get('jumlah_data_latih', '-')} apel\n"
            f"- Data uji: {paket.get('jumlah_data_uji', '-')} apel"
        )
        if metrik:
            tabel_metrik = pd.DataFrame(
                {
                    "Ukuran": ["Akurasi", "Precision", "Recall", "F1-score"],
                    "Nilai pada data uji": [
                        persen(metrik["Akurasi uji"]),
                        persen(metrik["Precision"]),
                        persen(metrik["Recall"]),
                        persen(metrik["F1-score"]),
                    ],
                }
            )
            st.table(tabel_metrik.set_index("Ukuran"))
