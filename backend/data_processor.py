import io
import os
import numpy as np
import pandas as pd
from typing import Dict, Any


class DataProcessor:
    CHUNK_SIZE = 10_000
    LARGE_THRESHOLD = 20_000
    MAX_CAT_VALUES = 10
    MAX_SAMPLE_ROWS = 10

    def process(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        ext = os.path.splitext(filename)[1].lower()
        if ext in (".xlsx", ".xls"):
            return self._process_excel(file_bytes, filename)
        elif ext in (".csv", ".tsv"):
            sep = "\t" if ext == ".tsv" else ","
            estimated = self._estimate_rows(file_bytes)
            if estimated > self.LARGE_THRESHOLD:
                return self._process_large_csv(file_bytes, filename, sep)
            else:
                try:
                    df = pd.read_csv(io.BytesIO(file_bytes), sep=sep)
                except Exception:
                    df = pd.read_csv(io.BytesIO(file_bytes), sep=sep, encoding="latin1")
                return self._from_df(df, filename)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _estimate_rows(self, file_bytes: bytes) -> int:
        try:
            sample = file_bytes[:10000].decode("utf-8", errors="ignore")
            n = sample.count("\n")
            return int(len(file_bytes) / (10000 / max(n, 1)))
        except Exception:
            return 0

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [str(c).strip() for c in df.columns]
        for col in df.select_dtypes(include="object").columns:
            try:
                parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                if parsed.notna().sum() > len(df) * 0.5:
                    df[col] = parsed
            except Exception:
                pass
        return df

    def _process_excel(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Read all sheets from an Excel file and build a combined summary."""
        sheets: Dict[str, pd.DataFrame] = pd.read_excel(
            io.BytesIO(file_bytes), sheet_name=None  # None = all sheets
        )
        sheet_names = list(sheets.keys())

        all_summaries = []
        total_rows = 0
        all_columns = {}

        for sheet_name, df in sheets.items():
            if df.empty:
                all_summaries.append(f"=== SHEET: '{sheet_name}' === (empty, skipped)")
                continue

            # Smart header detection: if all columns are "Unnamed", the sheet
            # likely has a styled title row. Scan the first 10 rows to find
            # the first row where most values are non-null strings → treat as header.
            if all(str(c).startswith("Unnamed:") for c in df.columns):
                header_row = self._detect_header_row(file_bytes, sheet_name)
                if header_row is not None:
                    df = pd.read_excel(
                        io.BytesIO(file_bytes),
                        sheet_name=sheet_name,
                        header=header_row
                    )
                    df.dropna(how="all", inplace=True)

            df = self._clean(df)
            total_rows += len(df)
            all_columns[sheet_name] = df.columns.tolist()
            sheet_summary = self._build_summary(df, f"{filename} > Sheet: '{sheet_name}'")
            all_summaries.append(sheet_summary)

        sep = "\n\n" + ("=" * 60) + "\n\n"
        combined_summary = (
            f"=== FILE: {filename} | {len(sheet_names)} sheet(s): "
            f"{', '.join(repr(s) for s in sheet_names)} ===\n\n"
            + sep.join(all_summaries)
        )

        meta = {
            "file_name": filename,
            "rows": total_rows,
            "columns": max((len(c) for c in all_columns.values()), default=0),
            "column_names": all_columns,   # dict: sheet_name -> [col, ...]
            "sheets": sheet_names,
            "sheet_count": len(sheet_names),
            "is_large": False,
        }
        return {"summary": combined_summary, "metadata": meta}

    def _detect_header_row(self, file_bytes: bytes, sheet_name: str):
        """Scan first 10 rows to find the first row that looks like a real header."""
        try:
            raw = pd.read_excel(
                io.BytesIO(file_bytes), sheet_name=sheet_name, header=None, nrows=10
            )
            for i, row in raw.iterrows():
                non_null = row.dropna()
                if len(non_null) >= 2:
                    str_count = sum(isinstance(v, str) for v in non_null)
                    if str_count / len(non_null) >= 0.6:
                        return i
        except Exception:
            pass
        return None

    def _from_df(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        df = self._clean(df)
        meta = {"file_name": filename, "rows": len(df), "columns": len(df.columns),
                "column_names": df.columns.tolist(), "is_large": False}
        stats = self._compute_stats(df)
        meta["stats"] = stats
        summary = self._build_summary(df, filename)
        return {"summary": summary, "metadata": meta}

    def _compute_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Return structured statistics for numeric, categorical and datetime columns.

        Numeric stats: count, sum, mean, median, std, min, max, 25pct,75pct
        Categorical stats: unique, top (list of (value,count))
        Datetime stats: min, max, count
        Also include up to MAX_SAMPLE_ROWS sample rows as list of dicts.
        """
        stats: Dict[str, Any] = {}
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols):
            num_stats = {}
            for col in num_cols:
                s = df[col].dropna()
                if len(s) == 0:
                    continue
                num_stats[col] = {
                    "count": int(s.count()),
                    "sum": float(s.sum()),
                    "mean": float(s.mean()),
                    "median": float(s.median()),
                    "std": float(s.std()),
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "25pct": float(s.quantile(0.25)),
                    "75pct": float(s.quantile(0.75)),
                }
            stats["numeric"] = num_stats

        cat_cols = df.select_dtypes(include=[object, "category"]).columns
        if len(cat_cols):
            cat_stats = {}
            for col in cat_cols:
                top = df[col].value_counts().head(self.MAX_CAT_VALUES)
                top_list = [(str(v), int(c)) for v, c in top.items()]
                cat_stats[col] = {"unique": int(df[col].nunique()), "missing": int(df[col].isna().sum()), "top": top_list}
            stats["categorical"] = cat_stats

        dt_cols = df.select_dtypes(include=["datetime64"]).columns
        if len(dt_cols):
            dt_stats = {}
            for col in dt_cols:
                s = df[col].dropna()
                dt_stats[col] = {"min": str(s.min()), "max": str(s.max()), "count": int(len(s))}
            stats["datetime"] = dt_stats

        # sample rows
        stats["samples"] = df.head(self.MAX_SAMPLE_ROWS).to_dict(orient="records")
        return stats

    def _build_summary(self, df: pd.DataFrame, filename: str) -> str:
        lines = [f"=== DATASET: {filename} ===",
                 f"Shape: {len(df):,} rows × {len(df.columns)} columns",
                 f"Columns: {', '.join(df.columns.tolist())}", ""]

        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols):
            lines.append("--- NUMERIC COLUMNS (Exact pre-computed statistics) ---")
            for col in num_cols:
                s = df[col].dropna()
                if len(s) == 0:
                    continue
                lines += [f"\n[{col}]",
                          f"  Count: {len(s):,} | Missing: {df[col].isna().sum():,}",
                          f"  Min: {s.min():,.4f} | Max: {s.max():,.4f}",
                          f"  Mean: {s.mean():,.4f} | Median: {s.median():,.4f}",
                          f"  Std: {s.std():,.4f} | Sum: {s.sum():,.4f}",
                          f"  25th pct: {s.quantile(0.25):,.4f} | 75th pct: {s.quantile(0.75):,.4f}"]

        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols):
            lines.append("\n--- CATEGORICAL COLUMNS ---")
            for col in cat_cols:
                top = df[col].value_counts().head(self.MAX_CAT_VALUES)
                top_str = ", ".join([f'"{v}" ({c:,})' for v, c in top.items()])
                lines += [f"\n[{col}]",
                          f"  Unique: {df[col].nunique():,} | Missing: {df[col].isna().sum():,}",
                          f"  Top values: {top_str}"]

        dt_cols = df.select_dtypes(include=["datetime64"]).columns
        if len(dt_cols):
            lines.append("\n--- DATETIME COLUMNS ---")
            for col in dt_cols:
                s = df[col].dropna()
                lines += [f"\n[{col}]", f"  Range: {s.min()} → {s.max()}",
                          f"  Count: {len(s):,} | Missing: {df[col].isna().sum():,}"]

        lines.append(f"\n--- SAMPLE ROWS (first {min(self.MAX_SAMPLE_ROWS, len(df))}) ---")
        lines.append(df.head(self.MAX_SAMPLE_ROWS).to_string(index=False))
        return "\n".join(lines)

    def _process_large_csv(self, file_bytes: bytes, filename: str, sep: str) -> Dict[str, Any]:
        """Stream large CSV in chunks, aggregate running stats."""
        first = pd.read_csv(io.BytesIO(file_bytes), sep=sep, nrows=100)
        first = self._clean(first)
        columns = first.columns.tolist()
        num_cols = first.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = first.select_dtypes(include=["object", "category"]).columns.tolist()

        stats = {c: {"n": 0, "s": 0.0, "ss": 0.0, "mn": np.inf, "mx": -np.inf}
                 for c in num_cols}
        cat_counts = {c: {} for c in cat_cols}
        total_rows = 0
        samples = []

        for i, chunk in enumerate(pd.read_csv(io.BytesIO(file_bytes), sep=sep,
                                               chunksize=self.CHUNK_SIZE)):
            chunk = self._clean(chunk)
            total_rows += len(chunk)
            if i < 3:
                samples.extend(chunk.head(3).to_dict("records"))

            for col in num_cols:
                if col in chunk.columns:
                    v = chunk[col].dropna()
                    if len(v):
                        stats[col]["n"] += len(v)
                        stats[col]["s"] += v.sum()
                        stats[col]["ss"] += (v ** 2).sum()
                        stats[col]["mn"] = min(stats[col]["mn"], v.min())
                        stats[col]["mx"] = max(stats[col]["mx"], v.max())

            for col in cat_cols:
                if col in chunk.columns:
                    for val, cnt in chunk[col].value_counts().head(50).items():
                        cat_counts[col][str(val)] = cat_counts[col].get(str(val), 0) + cnt

        # Build finalized stats
        num_stats = {}
        for col, s in stats.items():
            if s["n"]:
                mean = s["s"] / s["n"]
                std = np.sqrt(max(0, s["ss"] / s["n"] - mean ** 2))
                num_stats[col] = {"count": s["n"], "sum": round(s["s"], 4),
                                  "mean": round(mean, 4), "std": round(std, 4),
                                  "min": round(s["mn"], 4), "max": round(s["mx"], 4)}

        cat_stats = {c: {"unique": len(v),
                         "top": sorted(v.items(), key=lambda x: -x[1])[:self.MAX_CAT_VALUES]}
                     for c, v in cat_counts.items()}

        lines = [f"=== DATASET: {filename} (LARGE FILE — streamed) ===",
                 f"Shape: {total_rows:,} rows × {len(columns)} columns",
                 f"Columns: {', '.join(columns)}", ""]

        if num_stats:
            lines.append("--- NUMERIC COLUMNS (Aggregated from full dataset) ---")
            for col, s in num_stats.items():
                lines += [f"\n[{col}]",
                          f"  Count: {s['count']:,} | Sum: {s['sum']:,}",
                          f"  Min: {s['min']:,} | Max: {s['max']:,}",
                          f"  Mean: {s['mean']:,} | Std: {s['std']:,}"]

        if cat_stats:
            lines.append("\n--- CATEGORICAL COLUMNS ---")
            for col, s in cat_stats.items():
                top_str = ", ".join([f'"{v}" ({c:,})' for v, c in s["top"]])
                lines += [f"\n[{col}]", f"  Unique: {s['unique']:,}",
                          f"  Top values: {top_str}"]

        if samples:
            lines.append(f"\n--- SAMPLE ROWS ---")
            for i, row in enumerate(samples[:self.MAX_SAMPLE_ROWS]):
                lines.append(f"Row {i + 1}: {row}")

        meta = {"file_name": filename, "rows": total_rows, "columns": len(columns),
            "column_names": columns, "is_large": True}
        # attach computed stats
        meta["stats"] = {"numeric": num_stats, "categorical": cat_stats}
        return {"summary": "\n".join(lines), "metadata": meta}
