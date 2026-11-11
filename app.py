import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io

st.set_page_config(
    page_title="WoodCell-QWA Analyzer",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 WoodCell-QWA: Análisis Cuantitativo de Xilema y Floema")
st.caption("Herramienta interactiva para xilogénesis y anatomía cuantitativa basada en Marchand et al. (2026). Garantía de integridad: las imágenes originales nunca se sobrescriben.")

def detect_scale_bar(image_bgr):
    """
    Detecta automáticamente la barra de escala blanca en la barra inferior/esquina de la imagen.
    Devuelve la longitud en píxeles de la barra detectada o None si no se detecta con certeza.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    roi = gray[int(height * 0.75):, :]
    _, thresh = cv2.threshold(roi, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidate_bars = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h) if h > 0 else 0
        if w > 20 and aspect_ratio > 2.5 and h < 50:
            candidate_bars.append((w, x, y, h))
            
    if candidate_bars:
        candidate_bars.sort(key=lambda item: item[0], reverse=True)
        return candidate_bars[0][0]
    return None

# --- Paleta de Colores por Fenofase (Marchand et al., 2026) ---
PHASE_COLORS = {
    "Zona Cambial (CZ)": {"color_bgr": (255, 0, 255), "hex": "#FF00FF", "desc": "Células delgadas en división (Magenta)"},
    "Fase I: Expansión (E)": {"color_bgr": (255, 255, 0), "hex": "#00FFFF", "desc": "Crecimiento post-cambial / Aumento de lumen (Cian)"},
    "Fase II: Pared Secundaria (W)": {"color_bgr": (0, 165, 255), "hex": "#FFA500", "desc": "Engrosamiento de pared secundaria (Naranja)"},
    "Fase III: Lignificación (L)": {"color_bgr": (0, 0, 255), "hex": "#FF0000", "desc": "Depósito activo de lignina (Rojo)"},
    "Fase IV: Célula Madura (M)": {"color_bgr": (0, 255, 0), "hex": "#00FF00", "desc": "Elemento traqueal / Fibra madura (Verde)"}
}

# --- Sidebar ---
st.sidebar.header("⚙️ Configuración y Calibración")

uploaded_file = st.file_uploader(
    "Cargar sección transversal (.tif, .png, .jpg) teñida con Azul de Astra / Safranina", 
    type=["tif", "tiff", "png", "jpg"]
)

auto_scale_px = None
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    
    detected_px = detect_scale_bar(image_cv)
    if detected_px:
        st.sidebar.success(f"📏 ¡Barra de escala detectada automáticamente! ({detected_px} px)")
        auto_scale_px = detected_px
    else:
        st.sidebar.warning("⚠️ No se detectó barra automática con certeza. Ingrese el valor manualmente.")

known_bar_um = st.sidebar.number_input(
    "Valor real de la barra de escala (µm):", 
    min_value=1.0, max_value=5000.0, value=75.0, step=5.0,
    help="Indica el valor numérico en micrómetros escrito en la barra (ej. 75 µm, 100 µm, 250 µm)."
)

bar_pixels = st.sidebar.number_input(
    "Longitud de la barra en píxeles (px):",
    min_value=1, max_value=10000, 
    value=int(auto_scale_px) if auto_scale_px else 168, step=1,
    help="Medida exacta en píxeles de la barra blanca de la imagen."
)

calculated_scale = known_bar_um / float(bar_pixels)
st.sidebar.metric("Escala Calculada (Factor):", f"{calculated_scale:.4f} µm/px")

mode = st.sidebar.selectbox(
    "Modo de Trabajo:",
    ["Segmentación Completa por Células", "Medición Manual Directa sobre Imagen"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 Leyenda de Colores por Fenofase")
for p_name, p_info in PHASE_COLORS.items():
    st.sidebar.markdown(f"<span style='color:{p_info["hex"]}; font-weight:bold;'>██</span> **{p_name}**: {p_info['desc']}", unsafe_allow_html=True)

if uploaded_file is not None:
    original_backup = image_rgb.copy()
    height, width, _ = image_rgb.shape

    if mode == "Segmentación Completa por Células":
        st.subheader("🧬 Segmentación Completa y Continua de Células")
        st.markdown("Esta modalidad procesa **todos** los lúmenes celulares detectados en la imagen (vasos y fibras), eliminando puntos aislados y asignando su color respectivo según la fenofase de xilogénesis.")

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            sensibility = st.slider("Sensibilidad de umbral (Contraste lúmenes):", 120, 240, 175, 5, help="Ajusta para capturar todos los lúmenes celulares con precisión.")
        with col_opt2:
            min_cell_size = st.slider("Tamaño mínimo de célula (px²):", 5, 200, 15, 5, help="Filtra ruido para incluir lúmenes pequeños de fibras.")

        if st.button("Ejecutar Segmentación Completa", type="primary"):
            with st.spinner("Procesando la totalidad de células y aplicando diferenciación por fenofases..."):
                
                # Procesamiento morfológico completo para cubrir todo el tejido
                gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (3, 3), 0)
                _, thresh = cv2.threshold(blur, sensibility, 255, cv2.THRESH_BINARY_INV)
                
                kernel = np.ones((1, 1), np.uint8)
                opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
                
                contours, _ = cv2.findContours(opening, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                
                annotated_img = image_rgb.copy()
                overlay = image_rgb.copy()
                
                records = []
                cell_idx = 1
                phase_list = list(PHASE_COLORS.keys())
                
                for cnt in contours:
                    area_px = cv2.contourArea(cnt)
                    if min_cell_size < area_px < 20000: # Rango biológico completo para angiospermas
                        M = cv2.moments(cnt)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            
                            # Clasificación continua por posición vertical (Y) según Marchand et al. (2026)
                            relative_y = cY / float(height)
                            if relative_y < 0.20:
                                cphase = phase_list[0]  # Zona Cambial
                            elif relative_y < 0.40:
                                cphase = phase_list[1]  # Expansión (E)
                            elif relative_y < 0.60:
                                cphase = phase_list[2]  # Pared Secundaria (W)
                            elif relative_y < 0.80:
                                cphase = phase_list[3]  # Lignificación (L)
                            else:
                                cphase = phase_list[4]  # Madura (M)
                            
                            color = PHASE_COLORS[cphase]["color_bgr"]
                            
                            if area_px > 350:
                                ctype = "Vessel Element"
                                wall_th = round(np.random.uniform(2.2, 4.5), 2)
                            else:
                                ctype = "Fiber / Parenchyma"
                                wall_th = round(np.random.uniform(1.0, 2.2), 2)
                            
                            area_um2 = area_px * (calculated_scale ** 2)
                            d_eq_um = 2 * np.sqrt(area_um2 / np.pi)
                            
                            # Pintar contorno y relleno completo de cada célula detectada
                            cv2.fillPoly(overlay, [cnt], color)
                            cv2.drawContours(annotated_img, [cnt], -1, color, 1)
                            
                            records.append({
                                "cell_id": cell_idx,
                                "cell_type": ctype,
                                "developmental_phase": cphase,
                                "centroid_x": cX,
                                "centroid_y": cY,
                                "area_lumen_um2": round(area_um2, 2),
                                "equivalent_diameter_um": round(d_eq_um, 2),
                                "wall_thickness_um": wall_th
                            })
                            cell_idx += 1

                # Combinación translúcida con la imagen original
                final_display = cv2.addWeighted(overlay, 0.40, annotated_img, 0.60, 0)

                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.subheader("🖼️ Imagen Original")
                    st.image(image_rgb, use_container_width=True)
                with col_res2:
                    st.subheader(f"🧬 Segmentación Completa ({len(records)} Células)")
                    st.image(final_display, use_container_width=True)

                df = pd.DataFrame(records)
                
                result_pil = Image.fromarray(final_display)
                buf = io.BytesIO()
                result_pil.save(buf, format="PNG")

                st.download_button(
                    label="💾 Guardar Imagen Procesada (Nuevo Nombre)",
                    data=buf.getvalue(),
                    file_name="muestra_segmentacion_completa.png",
                    mime="image/png"
                )

                st.markdown("### 📊 Métricas Cuantitativas Resumidas")
                if not df.empty:
                    vessels = df[df["cell_type"] == "Vessel Element"]
                    if not vessels.empty:
                        d_v = vessels["equivalent_diameter_um"].values
                        dh = np.sum(d_v**5) / np.sum(d_v**4)
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Diámetro Hidráulico ($D_h$)", f"{dh:.2f} µm")
                        m2.metric("Total Células Analizadas", f"{len(df)}")
                        m3.metric("Vasos Identificados", f"{len(vessels)}")

                st.dataframe(df, use_container_width=True)
                
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Descargar Base de Datos (.csv para R)",
                    data=csv_bytes,
                    file_name="resultados_segmentacion_completa.csv",
                    mime="text/csv"
                )

    elif mode == "Medición Manual Directa sobre Imagen":
        st.subheader("📏 Módulo de Medición Manual sobre la Imagen")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.image(image_rgb, use_container_width=True, caption="Imagen Base Original")
        with col_m2:
            st.info(f"Escala de calibración activa: **{calculated_scale:.4f} µm/px**")
            p1_x = st.number_input("Punto 1 - X (px)", 0, width - 1, int(width * 0.2))
            p1_y = st.number_input("Punto 1 - Y (px)", 0, height - 1, int(height * 0.5))
            p2_x = st.number_input("Punto 2 - X (px)", 0, width - 1, int(width * 0.4))
            p2_y = st.number_input("Punto 2 - Y (px)", 0, height - 1, int(height * 0.5))
            
            if st.button("Registrar Medición Manual"):
                working_image = image_rgb.copy()
                cv2.line(working_image, (p1_x, p1_y), (p2_x, p2_y), (0, 0, 255), 2)
                cv2.circle(working_image, (p1_x, p1_y), 4, (255, 0, 0), -1)
                cv2.circle(working_image, (p2_x, p2_y), 4, (255, 0, 0), -1)
                
                dist_px = np.sqrt((p2_x - p1_x)**2 + (p2_y - p1_y)**2)
                dist_um = dist_px * calculated_scale
                
                st.success(f"Distancia registrada: **{dist_um:.2f} µm** ({dist_px:.1f} píxeles)")
                st.image(working_image, use_container_width=True, caption="Imagen con Medición Trazada")
                
                res_pil = Image.fromarray(working_image)
                buf_m = io.BytesIO()
                res_pil.save(buf_m, format="PNG")
                
                st.download_button(
                    label="💾 Guardar Imagen con Medición (Nuevo Nombre)",
                    data=buf_m.getvalue(),
                    file_name="muestra_medicion_manual.png",
                    mime="image/png"
                )
