"""
Genera ÚNICAMENTE los archivos .pkl y .faiss (con metadata) a partir de un CSV
usando embeddings servidos por Ollama. No hace búsquedas, ni chat, ni análisis.

Uso rápido:
  python generar_vectores_pickle_faiss.py \
      --csv partes_pc.csv \
      --out pc_parts_vectors \
      --model "gemma:2b" \
      --base-url http://localhost:11434 \
      --batch-size 16

El script guardará automáticamente los archivos en la carpeta ./vectores
  - vectores/pc_parts_vectors.pkl
  - vectores/pc_parts_vectors.faiss (si FAISS está disponible)
  - vectores/pc_parts_vectors_metadata.pkl
"""

import argparse
import os
import time
import pickle
import json
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import requests

# FAISS opcional
try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False


class Vectorizer:
    def __init__(
        self,
        csv_path: str,
        base_url: str = "http://localhost:11434",
        model: str = "gemma:2b",
        embedding_columns: List[str] | None = None,
        metadata_columns: List[str] | None = None,
        include_all_columns: bool = False,
        batch_size: int = 16,
        timeout: int = 30,
        out_dir: str = "vectores",
    ) -> None:
        self.csv_path = csv_path
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.embedding_columns = embedding_columns or ["CATEGORIA", "PRODUCTO"]
        self.metadata_columns = metadata_columns or ["COD", "PRECIO_MILES"]
        self.include_all_columns = include_all_columns
        self.batch_size = max(1, batch_size)
        self.timeout = timeout
        self.out_dir = out_dir

        self.df: pd.DataFrame | None = None
        self.texts: list[str] = []
        self.embeddings_matrix: np.ndarray | None = None
        self.faiss_index = None

        # Crear carpeta de salida si no existe
        os.makedirs(self.out_dir, exist_ok=True)

    # ---- Carga y preparación ----
    def load_csv(self) -> None:
        self.df = pd.read_csv(self.csv_path)
        # Validación básica
        missing = [c for c in self.embedding_columns if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"Columnas faltantes para embeddings: {missing}. Presentes: {list(self.df.columns)}"
            )

    def build_texts(self) -> None:
        assert self.df is not None
        cols_to_use = (
            self.df.columns.tolist() if self.include_all_columns else self.embedding_columns + self.metadata_columns
        )
        texts: list[str] = []
        for _, row in self.df.iterrows():
            parts: list[str] = []
            # columnas principales
            for c in self.embedding_columns:
                if c in self.df.columns and pd.notna(row[c]):
                    parts.append(f"{c}: {row[c]}")
            # metadata
            for c in self.metadata_columns:
                if c in self.df.columns and pd.notna(row[c]):
                    if c == "PRECIO_MILES":
                        parts.append(f"Precio: ${row[c]}k")
                    else:
                        parts.append(f"{c}: {row[c]}")
            # resto si aplica
            if self.include_all_columns:
                other = [c for c in self.df.columns if c not in self.embedding_columns + self.metadata_columns]
                for c in other:
                    if pd.notna(row[c]):
                        parts.append(f"{c}: {row[c]}")
            texts.append(" | ".join(parts))
        self.texts = texts

    # ---- Embeddings ----
    def _get_embedding(self, text: str) -> list[float]:
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.model, "prompt": text}
        r = requests.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        # Ollama responde {"embedding": [...]} 
        return data["embedding"]

    def vectorize(self) -> None:
        if not self.texts:
            self.build_texts()
        embs: list[list[float]] = []
        total = len(self.texts)
        for i in range(0, total, self.batch_size):
            batch = self.texts[i : i + self.batch_size]
            for t in batch:
                try:
                    embs.append(self._get_embedding(t))
                except requests.exceptions.RequestException as e:
                    raise RuntimeError(f"Error al obtener embedding: {e}") from e
            # micro pausa para no saturar
            if i + self.batch_size < total:
                time.sleep(0.4)
        self.embeddings_matrix = np.array(embs, dtype=np.float32)

    # ---- FAISS ----
    def build_faiss(self) -> None:
        if not FAISS_AVAILABLE:
            return
        assert self.embeddings_matrix is not None
        dim = self.embeddings_matrix.shape[1]
        index = faiss.IndexFlatIP(dim)
        vecs = self.embeddings_matrix.copy()
        faiss.normalize_L2(vecs)
        index.add(vecs)
        self.faiss_index = index

    # ---- Guardado ----
    def save_outputs(self, out_base: str) -> list[str]:
        assert self.df is not None and self.embeddings_matrix is not None
        saved: list[str] = []

        base_path = os.path.join(self.out_dir, out_base)

        # PKL completo (incluye embeddings_matrix como lista para portabilidad)
        pkl_data = {
            "df": self.df,
            "product_texts": self.texts,
            "embeddings_matrix": self.embeddings_matrix.tolist(),
            "embedding_columns": self.embedding_columns,
            "metadata_columns": self.metadata_columns,
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
        }
        pkl_file = f"{base_path}.pkl"
        with open(pkl_file, "wb") as f:
            pickle.dump(pkl_data, f)
        saved.append(pkl_file)

        # FAISS + metadata (si FAISS está disponible)
        if FAISS_AVAILABLE and self.faiss_index is not None:
            faiss_file = f"{base_path}.faiss"
            meta = {
                "df": self.df,
                "product_texts": self.texts,
                "embedding_columns": self.embedding_columns,
                "metadata_columns": self.metadata_columns,
                "timestamp": datetime.now().isoformat(),
                "model": self.model,
            }
            meta_file = f"{base_path}_metadata.pkl"
            faiss.write_index(self.faiss_index, faiss_file)
            with open(meta_file, "wb") as f:
                pickle.dump(meta, f)
            saved.extend([faiss_file, meta_file])
        return saved


def main() -> None:
    ap = argparse.ArgumentParser(description="Generar únicamente PKL y FAISS a partir de un CSV")
    ap.add_argument("--csv", required=True, help="Ruta al CSV de productos")
    ap.add_argument("--out", default="pc_parts_vectors", help="Base del nombre de salida (sin extensión)")
    ap.add_argument("--model", default="gemma:2b", help="Modelo de embeddings en Ollama")
    ap.add_argument("--base-url", default="http://localhost:11434", help="URL base de Ollama")
    ap.add_argument("--batch-size", type=int, default=16, help="Tamaño de lote para solicitar embeddings")
    ap.add_argument("--all", action="store_true", help="Usar TODAS las columnas del CSV")
    ap.add_argument(
        "--embed-cols",
        nargs="*",
        default=["CATEGORIA", "PRODUCTO"],
        help="Columnas para construir el texto de embeddings (ignorado si --all)",
    )
    ap.add_argument(
        "--meta-cols",
        nargs="*",
        default=["COD", "PRECIO_MILES"],
        help="Columnas de metadata para el texto (ignorado si --all)",
    )
    args = ap.parse_args()

    v = Vectorizer(
        csv_path=args.csv,
        base_url=args.base_url,
        model=args.model,
        embedding_columns=args.embed_cols,
        metadata_columns=args.meta_cols,
        include_all_columns=args.all,
        batch_size=args.batch_size,
        out_dir="vectores",
    )

    print("📥 Cargando CSV...")
    v.load_csv()
    print("🧾 Construyendo textos...")
    v.build_texts()
    print(f"🧠 Generando embeddings ({len(v.texts)} items)...")
    v.vectorize()

    if FAISS_AVAILABLE:
        print("🧱 Construyendo índice FAISS...")
        v.build_faiss()
    else:
        print("⚠️ FAISS no disponible; se generará solo el .pkl (y _metadata.pkl no tendrá índice)")

    print("💾 Guardando salidas en ./vectores ...")
    files = v.save_outputs(args.out)
    for f in files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  • {f} ({size_mb:.2f} MB)")
    print("✅ Listo.")


if __name__ == "__main__":
    import sys
    
    # Detectar si se está ejecutando en Jupyter/IPython
    if "ipykernel_launcher" in sys.argv[0]:
        # Simular parámetros por defecto
        sys.argv = [
            sys.argv[0],
            "--csv", "partes_pc.csv",
            "--out", "pc_parts_vectors",
            "--model", "gemma:2b",
            "--base-url", "http://localhost:11434",
            "--batch-size", "16"
        ]

    elif len(sys.argv) == 1:
        # Caso normal: sin argumentos en terminal
        sys.argv.extend([
            "--csv", "partes_pc.csv",
            "--out", "pc_parts_vectors",
            "--model", "gemma:2b",
            "--base-url", "http://localhost:11434",
            "--batch-size", "16"
        ])

    main()