import io
import os
import re
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

log = logging.getLogger("data_processor")

# ── Keywords that identify "summary / total" rows ────────────────────────────
TOTAL_KEYWORDS = re.compile(
    r"\b(total|grand total|subtotal|sum|efficiency|average|avg|growth|manufactured|work points)\b",
    re.IGNORECASE,
)

# ── Keywords for employee name columns ──────────────────────────────────────
EMPLOYEE_COL_KEYWORDS = re.compile(
    r"\b(employee|name|staff|worker|person|operator|technician|agent)\b",
    re.IGNORECASE,
)


class DataProcessor:
    CHUNK_SIZE = 10_000
    LARGE_THRESHOLD = 20_000
    MAX_CAT_VALUES = 10
    MAX_SAMPLE_ROWS = 10
    _cache: Dict[str, Any] = {}   # P4.1 — in-memory hash cache

    # ── Public API ────────────────────────────────────────────────────────────

    def process_cached(self, file_bytes: bytes, filename: str, file_hash: str) -> Dict[str, Any]:
        """Return cached result if the same file was processed before (P4.1)."""
        if file_hash in self._cache:
            log.info("Cache hit for hash %s", file_hash[:12])
            return self._cache[file_hash]
        result = self.process(file_bytes, filename)
        self._cache[file_hash] = result
        return result

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

    # ── Internal helpers ──────────────────────────────────────────────────────

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

    # ── P0.2: Named totals extractor ─────────────────────────────────────────

    def _extract_named_totals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Scan every row for cells matching TOTAL_KEYWORDS in any text column.
        If a matching label is found in the same row as a numeric value, record it.
        Returns a dict like {"TOTAL POINTS": 31735, "EFFICIENCY": 0.383, ...}
        """
        totals: Dict[str, Any] = {}
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = df.select_dtypes(include=["object"]).columns.tolist()

        for _, row in df.iterrows():
            # Look for a text cell that matches keyword pattern
            label = None
            for tc in text_cols:
                val = str(row.get(tc, "")).strip()
                if val and TOTAL_KEYWORDS.search(val):
                    label = val
                    break

            if label is None:
                continue

            # Collect numeric values from same row
            nums = {}
            for nc in num_cols:
                v = row.get(nc)
                if pd.notna(v):
                    nums[str(nc).strip()] = float(v)

            # If only one numeric → use label as key directly
            if len(nums) == 1:
                val = list(nums.values())[0]
                totals[label.upper()] = val
            elif len(nums) > 1:
                # Multiple numerics — store each as LABEL:COLUMN
                for nc_name, val in nums.items():
                    key = f"{label.upper()} [{nc_name}]"
                    totals[key] = val

        return totals

    # ── P1.1: Employee stats engine ──────────────────────────────────────────

    def _detect_employee_column(self, df: pd.DataFrame) -> Optional[str]:
        """Return the best-guess employee name column, or None."""
        for col in df.columns:
            if EMPLOYEE_COL_KEYWORDS.search(str(col)):
                return col
        # fallback: look for a string column whose values are predominantly
        # Title-Case words (likely person names), e.g. "ROHIT PATOLIYA"
        text_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for col in text_cols:
            sample = df[col].dropna().head(20)
            title_like = sum(
                1 for v in sample
                if isinstance(v, str) and len(v.split()) >= 2 and v == v.upper()
            )
            if title_like >= max(2, len(sample) * 0.4):
                return col
        return None

    def _extract_employee_stats(
        self, sheets: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, Any]]:
        """
        P1.1 + P1.2 — For each sheet, detect the employee column, group
        numeric columns by employee, then cross-stitch all sheets into a
        unified per-employee record.
        """
        per_sheet: Dict[str, Dict[str, Dict[str, float]]] = {}   # sheet → emp → {col: val}

        for sheet_name, df in sheets.items():
            if df.empty:
                continue
            emp_col = self._detect_employee_column(df)
            if emp_col is None:
                continue

            num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                        if c != emp_col]
            if not num_cols:
                continue

            grouped = df.groupby(emp_col, as_index=False)[num_cols].sum(numeric_only=True)
            sheet_emp_stats: Dict[str, Dict[str, float]] = {}
            for _, row in grouped.iterrows():
                emp = str(row[emp_col]).strip()
                if not emp or emp.lower() in ("nan", "total", "grand total"):
                    continue
                sheet_emp_stats[emp] = {
                    col: float(row[col]) for col in num_cols if pd.notna(row.get(col))
                }
            if sheet_emp_stats:
                per_sheet[sheet_name] = sheet_emp_stats

        if not per_sheet:
            return {}

        # Cross-stitch: merge all sheets into one record per employee
        all_employees: Dict[str, Dict[str, Any]] = {}
        for sheet_name, emp_dict in per_sheet.items():
            for emp, stats in emp_dict.items():
                if emp not in all_employees:
                    all_employees[emp] = {}
                for col, val in stats.items():
                    key = f"{sheet_name}_{col}"
                    all_employees[emp][key] = val

        # Compute overall rank by total numeric value per employee
        totals_by_emp = {
            emp: sum(v for v in stats.values() if isinstance(v, (int, float)))
            for emp, stats in all_employees.items()
        }
        sorted_emps = sorted(totals_by_emp, key=lambda e: totals_by_emp[e], reverse=True)
        for rank, emp in enumerate(sorted_emps, 1):
            all_employees[emp]["overall_rank"] = rank
            all_employees[emp]["overall_total"] = totals_by_emp[emp]

        return all_employees

    # ── Excel processing ──────────────────────────────────────────────────────

    def _process_excel(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Read all sheets, build summary, extract named totals + employee stats."""
        raw_sheets: Dict[str, pd.DataFrame] = pd.read_excel(
            io.BytesIO(file_bytes), sheet_name=None
        )
        sheet_names = list(raw_sheets.keys())

        # Clean & detect headers
        cleaned_sheets: Dict[str, pd.DataFrame] = {}
        for sheet_name, df in raw_sheets.items():
            if df.empty:
                cleaned_sheets[sheet_name] = df
                continue
            if all(str(c).startswith("Unnamed:") for c in df.columns):
                header_row = self._detect_header_row(file_bytes, sheet_name)
                if header_row is not None:
                    df = pd.read_excel(
                        io.BytesIO(file_bytes),
                        sheet_name=sheet_name,
                        header=header_row,
                    )
                    df.dropna(how="all", inplace=True)
            df = self._clean(df)
            cleaned_sheets[sheet_name] = df

        # ── Named totals (P0.2) ───────────────────────────────────────────
        all_named_totals: Dict[str, Any] = {}
        for sheet_name, df in cleaned_sheets.items():
            if df.empty:
                continue
            sheet_totals = self._extract_named_totals(df)
            for k, v in sheet_totals.items():
                all_named_totals[k] = v

        # ── Employee stats (P1.1 + P1.2) ─────────────────────────────────
        employee_stats = self._extract_employee_stats(cleaned_sheets)

        # ── Per-sheet summaries + numeric stats ──────────────────────────
        all_summaries: List[str] = []
        all_columns: Dict[str, List[str]] = {}
        combined_numeric_stats: Dict[str, Any] = {}
        combined_categorical_stats: Dict[str, Any] = {}
        total_rows = 0
        dataframe_json = None   # only for single-sheet / first sheet

        for sheet_name, df in cleaned_sheets.items():
            if df.empty:
                all_summaries.append(f"=== SHEET: '{sheet_name}' === (empty, skipped)")
                continue
            total_rows += len(df)
            all_columns[sheet_name] = df.columns.tolist()

            sheet_stats = self._compute_stats(df)
            combined_numeric_stats[sheet_name] = sheet_stats.get("numeric", {})
            combined_categorical_stats[sheet_name] = sheet_stats.get("categorical", {})

            # Build text summary for this sheet
            sheet_summary = self._build_summary(
                df, f"{filename} > Sheet: '{sheet_name}'",
                named_totals=self._extract_named_totals(df),
            )
            all_summaries.append(sheet_summary)

            # Store first sheet's DataFrame as JSON for /filter
            if dataframe_json is None:
                try:
                    dataframe_json = df.head(50_000).to_json(
                        orient="split", date_format="iso", default_handler=str
                    )
                except Exception:
                    dataframe_json = None

        sep = "\n\n" + ("=" * 60) + "\n\n"
        combined_summary = (
            f"=== FILE: {filename} | {len(sheet_names)} sheet(s): "
            f"{', '.join(repr(s) for s in sheet_names)} ===\n\n"
            + sep.join(all_summaries)
        )

        # Append named totals block to summary (P2.3)
        if all_named_totals:
            combined_summary += "\n\n" + self._format_named_totals_block(all_named_totals)

        # Append employee stats block to summary (P1.3)
        if employee_stats:
            combined_summary += "\n\n" + self._format_employee_stats_block(employee_stats)

        meta = {
            "file_name": filename,
            "rows": total_rows,
            "columns": max((len(c) for c in all_columns.values()), default=0),
            "column_names": all_columns,
            "sheets": sheet_names,
            "sheet_count": len(sheet_names),
            "is_large": False,
            "stats": {
                "numeric": combined_numeric_stats,
                "categorical": combined_categorical_stats,
            },
            "named_totals": all_named_totals,
            "employee_stats": employee_stats,
        }

        return {
            "summary": combined_summary,
            "metadata": meta,
            "dataframe_json": dataframe_json,
            # P0 — store raw DataFrames (serialised per sheet) for pandas compute
            "sheets_data": {
                name: df.to_json(orient="split", date_format="iso", default_handler=str)
                for name, df in cleaned_sheets.items()
                if not df.empty
            },
        }

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

    # ── Single-DF path (CSV) ──────────────────────────────────────────────────

    def _from_df(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        df = self._clean(df)
        named_totals = self._extract_named_totals(df)
        stats = self._compute_stats(df)
        meta = {
            "file_name": filename,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "is_large": False,
            "stats": stats,
            "named_totals": named_totals,
            "employee_stats": {},
        }
        summary = self._build_summary(df, filename, named_totals=named_totals)
        if named_totals:
            summary += "\n\n" + self._format_named_totals_block(named_totals)
        df_for_filter = df.head(50_000)
        try:
            dataframe_json = df_for_filter.to_json(orient="split", date_format="iso", default_handler=str)
        except Exception:
            dataframe_json = None
        return {"summary": summary, "metadata": meta, "dataframe_json": dataframe_json}

    # ── Stats helpers ─────────────────────────────────────────────────────────

    def _compute_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Return structured statistics for numeric, categorical and datetime columns."""
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
                cat_stats[col] = {
                    "unique": int(df[col].nunique()),
                    "missing": int(df[col].isna().sum()),
                    "top": top_list,
                }
            stats["categorical"] = cat_stats

        dt_cols = df.select_dtypes(include=["datetime64"]).columns
        if len(dt_cols):
            dt_stats = {}
            for col in dt_cols:
                s = df[col].dropna()
                dt_stats[col] = {"min": str(s.min()), "max": str(s.max()), "count": int(len(s))}
            stats["datetime"] = dt_stats

        stats["samples"] = df.head(self.MAX_SAMPLE_ROWS).to_dict(orient="records")
        return stats

    def _build_summary(
        self,
        df: pd.DataFrame,
        filename: str,
        named_totals: Optional[Dict[str, Any]] = None,
    ) -> str:
        lines = [
            f"=== DATASET: {filename} ===",
            f"Shape: {len(df):,} rows × {len(df.columns)} columns",
            f"Columns: {', '.join(df.columns.tolist())}",
            "",
        ]

        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols):
            lines.append("--- NUMERIC COLUMNS (Exact pre-computed statistics) ---")
            for col in num_cols:
                s = df[col].dropna()
                if len(s) == 0:
                    continue
                lines += [
                    f"\n[{col}]",
                    f"  Count: {len(s):,} | Missing: {df[col].isna().sum():,}",
                    f"  Min: {s.min():,.4f} | Max: {s.max():,.4f}",
                    f"  Mean: {s.mean():,.4f} | Median: {s.median():,.4f}",
                    f"  Std: {s.std():,.4f} | Sum: {s.sum():,.4f}",
                    f"  25th pct: {s.quantile(0.25):,.4f} | 75th pct: {s.quantile(0.75):,.4f}",
                ]

        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols):
            lines.append("\n--- CATEGORICAL COLUMNS ---")
            for col in cat_cols:
                top = df[col].value_counts().head(self.MAX_CAT_VALUES)
                top_str = ", ".join([f'"{v}" ({c:,})' for v, c in top.items()])
                lines += [
                    f"\n[{col}]",
                    f"  Unique: {df[col].nunique():,} | Missing: {df[col].isna().sum():,}",
                    f"  Top values: {top_str}",
                ]

        dt_cols = df.select_dtypes(include=["datetime64"]).columns
        if len(dt_cols):
            lines.append("\n--- DATETIME COLUMNS ---")
            for col in dt_cols:
                s = df[col].dropna()
                lines += [
                    f"\n[{col}]",
                    f"  Range: {s.min()} → {s.max()}",
                    f"  Count: {len(s):,} | Missing: {df[col].isna().sum():,}",
                ]

        lines.append(f"\n--- SAMPLE ROWS (first {min(self.MAX_SAMPLE_ROWS, len(df))}) ---")
        lines.append(df.head(self.MAX_SAMPLE_ROWS).to_string(index=False))
        return "\n".join(lines)

    # ── P2.3: Structured fact blocks injected into the AI context ────────────

    @staticmethod
    def _format_named_totals_block(totals: Dict[str, Any]) -> str:
        if not totals:
            return ""
        lines = [
            "[COMPUTED FACTS — USE THESE EXACTLY, DO NOT RECALCULATE]",
            "The following values are pre-calculated in the source file:",
        ]
        for k, v in totals.items():
            if isinstance(v, float) and v != int(v):
                lines.append(f"  {k}: {v:,.4f}")
            else:
                lines.append(f"  {k}: {v:,.0f}" if isinstance(v, (int, float)) else f"  {k}: {v}")
        return "\n".join(lines)

    @staticmethod
    def _format_employee_stats_block(stats: Dict[str, Dict[str, Any]]) -> str:
        if not stats:
            return ""
        lines = [
            "[EMPLOYEE / PERSON STATISTICS — USE THESE EXACTLY FOR PERFORMANCE QUESTIONS]",
        ]
        for emp, emp_stats in sorted(stats.items(),
                                     key=lambda x: x[1].get("overall_rank", 999)):
            rank = emp_stats.get("overall_rank", "?")
            total = emp_stats.get("overall_total", 0)
            lines.append(f"\n  {emp} (Overall Rank #{rank}, Total Score: {total:,.2f})")
            for k, v in emp_stats.items():
                if k in ("overall_rank", "overall_total"):
                    continue
                lines.append(f"    {k}: {v:,.2f}" if isinstance(v, float) else f"    {k}: {v}")
        return "\n".join(lines)

    # ── Large CSV (unchanged logic, minor cleanup) ────────────────────────────

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

        num_stats = {}
        for col, s in stats.items():
            if s["n"]:
                mean = s["s"] / s["n"]
                std = np.sqrt(max(0, s["ss"] / s["n"] - mean ** 2))
                num_stats[col] = {
                    "count": s["n"], "sum": round(s["s"], 4),
                    "mean": round(mean, 4), "std": round(std, 4),
                    "min": round(s["mn"], 4), "max": round(s["mx"], 4),
                }

        cat_stats = {
            c: {"unique": len(v),
                "top": sorted(v.items(), key=lambda x: -x[1])[:self.MAX_CAT_VALUES]}
            for c, v in cat_counts.items()
        }

        lines = [
            f"=== DATASET: {filename} (LARGE FILE — streamed) ===",
            f"Shape: {total_rows:,} rows × {len(columns)} columns",
            f"Columns: {', '.join(columns)}", "",
        ]
        if num_stats:
            lines.append("--- NUMERIC COLUMNS (Aggregated from full dataset) ---")
            for col, s in num_stats.items():
                lines += [
                    f"\n[{col}]",
                    f"  Count: {s['count']:,} | Sum: {s['sum']:,}",
                    f"  Min: {s['min']:,} | Max: {s['max']:,}",
                    f"  Mean: {s['mean']:,} | Std: {s['std']:,}",
                ]
        if cat_stats:
            lines.append("\n--- CATEGORICAL COLUMNS ---")
            for col, s in cat_stats.items():
                top_str = ", ".join([f'"{v}" ({c:,})' for v, c in s["top"]])
                lines += [f"\n[{col}]", f"  Unique: {s['unique']:,}", f"  Top values: {top_str}"]
        if samples:
            lines.append(f"\n--- SAMPLE ROWS ---")
            for i, row in enumerate(samples[:self.MAX_SAMPLE_ROWS]):
                lines.append(f"Row {i + 1}: {row}")

        meta = {
            "file_name": filename,
            "rows": total_rows,
            "columns": len(columns),
            "column_names": columns,
            "is_large": True,
            "stats": {"numeric": num_stats, "categorical": cat_stats},
            "named_totals": {},
            "employee_stats": {},
        }
        return {"summary": "\n".join(lines), "metadata": meta}
