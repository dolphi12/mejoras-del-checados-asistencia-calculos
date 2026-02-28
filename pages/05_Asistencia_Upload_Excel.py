"""
Streamlit page: Asistencia (Upload Excel)
---------------------------------------

This page allows users to upload an attendance export in Excel format and
calculate derived metrics (descuento no laborado, tiempo trabajado,
horas extra) based on a simplified algorithm inspired by the React
prototype provided by the user.  Unlike the main "Cálculo de Asistencia"
page, this page does not rely on the legacy event sourcing logic;
instead it assigns events by count and provides a fixed rule for lunch
deduction.  Users can adjust the base work day (in hours) before
processing and may edit the resulting table prior to export.

The workflow is:

1. Select an Excel file (``.xlsx``) containing a sheet named
   ``JORNADAS_CIERRE``.
2. Optionally adjust the jornada base (hours); defaults to 8h (480 min).
3. The file is processed and a table appears.  Columns for events and
   permission minutes are editable; computed columns update after edits.
4. Use the "Exportar Excel" button to download the final table.

Note: This page is available only to authenticated users.  It does not
persist any overrides and operates entirely client-side during the
session.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from typing import Optional

import streamlit as st
import pandas as pd

from components.layout import render_topbar
from components.auth_ui import login_panel, current_user
from components.sidebar_settings import render_sidebar_settings
from utils.page_bootstrap import bootstrap_page
from utils.runtime_store import get_data_dir
from utils.attendance_upload_excel import parse_upload_excel, recalc_upload_table


def _make_temp_copy(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> str:
    """Write an UploadedFile to a temporary file and return its path."""
    suffix = ".xlsx"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def main():  # pragma: no cover - Streamlit entry point isn't exercised by tests
    bootstrap_page("Asistencia (Upload Excel) | URSOMEX", page_icon="📁")
    # Sidebar settings such as dark mode toggle
    render_sidebar_settings()
    data_dir = get_data_dir()
    login_panel(data_dir)
    user = current_user()
    if not user:
        st.warning("Inicia sesión para usar esta sección.")
        return

    render_topbar("Asistencia (Upload Excel)", "Sube un Excel exportado por el Colector y calcula la asistencia con reglas simplificadas.")

    # Configuración
    st.sidebar.markdown("### Configuración")
    jornada_horas = st.sidebar.number_input(
        "Jornada base (horas)", min_value=0.0, max_value=24.0, value=8.0, step=0.5,
        help="Duración de la jornada laboral base. Se resta de las horas trabajadas para calcular horas extra."
    )
    jornada_base = int(jornada_horas * 60)

    st.sidebar.markdown("### Tolerancia de comida")
    # Estos parámetros se ajustan por exportación (diaria). No se redondea.
    lunch_expected = st.sidebar.number_input(
        "Duración esperada comida (min)", min_value=0, max_value=240, value=60, step=1,
        help="Duración objetivo de comida para aplicar descuento fijo."
    )
    lunch_fixed_discount = st.sidebar.number_input(
        "Descuento fijo si cae en tolerancia (min)", min_value=0, max_value=240, value=30, step=1,
        help="Si la comida cae dentro del rango de tolerancia, se descuenta este valor fijo."
    )
    ctol1, ctol2 = st.sidebar.columns(2)
    with ctol1:
        lunch_tol_minus = st.number_input(
            "Tolerancia - (min)", min_value=0, max_value=120, value=0, step=1,
            help="Resta a la duración esperada para definir el mínimo aceptado."
        )
    with ctol2:
        lunch_tol_plus = st.number_input(
            "Tolerancia + (min)", min_value=0, max_value=120, value=0, step=1,
            help="Suma a la duración esperada para definir el máximo aceptado."
        )

    uploaded_file: Optional[st.runtime.uploaded_file_manager.UploadedFile] = st.file_uploader(
        "Selecciona un archivo Excel (hoja JORNADAS_CIERRE)", type=["xlsx"]
    )
    if not uploaded_file:
        st.info("No se ha seleccionado ningún archivo.")
        return

    # Process file once uploaded
    tmp_path = _make_temp_copy(uploaded_file)
    try:
        df = parse_upload_excel(
            tmp_path,
            jornada_base=jornada_base,
            lunch_expected_minutes=int(lunch_expected),
            lunch_discount_minutes=int(lunch_fixed_discount),
            lunch_tolerance_minus=int(lunch_tol_minus),
            lunch_tolerance_plus=int(lunch_tol_plus),
        )
    except RuntimeError as exc:
        st.error(str(exc))
        return
    finally:
        # Clean up temporary file
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    if df is None or df.empty:
        st.warning("No se encontraron registros en la hoja JORNADAS_CIERRE.")
        return

    st.caption(
        "Puedes editar las columnas de tiempo (HH:MM) y permiso antes de exportar. "
        "Las columnas calculadas se actualizarán automáticamente."
    )

    # Keep editable copy in session state to persist edits across reruns
    key = "upload_attendance_df"
    if key not in st.session_state:
        st.session_state[key] = df.copy()
    # If dataset shape changes (e.g., new file), update session copy
    # También refrescamos cuando cambia el contenido (no solo la forma),
    # por ejemplo al subir otro archivo con el mismo número de filas/columnas.
    if st.session_state[key].shape != df.shape or not st.session_state[key].equals(df):
        st.session_state[key] = df.copy()

    # Display editor (Streamlit built-in).  Mark computed columns as read-only.
    editable_cols = [
        "ENTRADA",
        "SALIDA A COMER",
        "REGRESO DE COMER",
        "SALIDA A CENAR",
        "REGRESO DE CENAR",
        "SALIDA",
        "PERMISO (min)",
    ]
    disabled_cols = [c for c in df.columns if c not in editable_cols]

    edited_df = st.data_editor(
        st.session_state[key],
        key="upload_editor",
        num_rows="dynamic",
        hide_index=True,
        column_config={
            "PERMISO (min)": st.column_config.NumberColumn("PERMISO (min)", min_value=0, step=1),
        },
        disabled=disabled_cols,
        width="stretch",
    )

    # Recalcular con la tolerancia del día (configurable en sidebar)
    result_df = recalc_upload_table(
        edited_df,
        jornada_base=jornada_base,
        lunch_expected_minutes=int(lunch_expected),
        lunch_discount_minutes=int(lunch_fixed_discount),
        lunch_tolerance_minus=int(lunch_tol_minus),
        lunch_tolerance_plus=int(lunch_tol_plus),
    )

    # Persist the edited (but not recalculated) version for next run
    st.session_state[key] = edited_df.copy()

    # Display recalculated metrics alongside
    st.subheader("Resumen recalcualdo")
    # Render the recalculated table full width using the new width API
    st.dataframe(result_df, width="stretch")

    # Export button: generate Excel from recalculated DataFrame
    if st.button("Exportar Excel", type="primary"):
        # Save DataFrame to a temporary file
        out_dir = tempfile.mkdtemp(prefix="asist_upload_export_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"asistencia_calculada_{timestamp}.xlsx")
        # Convert numeric NaNs to zeros for Excel
        df_to_write = result_df.copy()
        numeric_cols = ["PERMISO (min)", "DESCUENTO NO LABORADO (min)", "TIEMPO TRABAJADO (min)", "HORAS EXTRA (min)"]
        for c in numeric_cols:
            if c in df_to_write.columns:
                df_to_write[c] = pd.to_numeric(df_to_write[c], errors="coerce").fillna(0)
        df_to_write.to_excel(out_path, index=False)
        with open(out_path, "rb") as f:
            st.download_button(
                label="Descargar Excel", data=f, file_name=os.path.basename(out_path), mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="download_excel_upload"
            )
        # Remove temp file after offering download (the file may still be open by browser)
        # Clean-up is deferred since Streamlit will stream the data immediately
