from .kernel_manager import ExecutionResult

MAX_OUTPUT_CHARS = 3000
MAX_DATAFRAME_ROWS = 5
MAX_TRACEBACK_LINES = 3


class OutputCompressor:
    """将沙箱执行结果压缩为 LLM 友好摘要，防止上下文爆炸。"""

    def compress(self, result: ExecutionResult) -> str:
        parts = []

        if result.text.strip():
            compressed_text = self._compress_text(result.text)
            parts.append(compressed_text)

        if result.images:
            parts.append(f"\n[生成 {len(result.images)} 张图表]")
            for img_path in result.images:
                parts.append(f"  ![](data/images/{img_path.split(chr(92))[-1].split('/')[-1]})")

        if result.error:
            parts.append(f"\n[执行错误] {result.error}")

        if not parts:
            parts.append("[代码执行完成，无输出]")

        return "\n".join(parts)

    def _compress_text(self, text: str) -> str:
        if len(text) <= MAX_OUTPUT_CHARS:
            return text

        lines = text.split("\n")
        if self._is_dataframe_output(lines):
            return self._compress_df_output(lines)

        if self._is_traceback(lines):
            return self._compress_traceback(text)

        # Generic: truncate from middle
        return self._truncate_middle(text)

    def _is_dataframe_output(self, lines: list[str]) -> bool:
        # Don't mis-classify tracebacks as DataFrames
        full = "\n".join(lines)
        if "Traceback" in full:
            return False
        # Heuristic: lots of whitespace-aligned columns
        aligned_cols = sum(1 for l in lines if "  " in l and l.strip())
        return aligned_cols > 5

    def _compress_df_output(self, lines: list[str]) -> str:
        header = lines[0] if lines else ""
        n_rows = len(lines) - 1
        preview = "\n".join(lines[: MAX_DATAFRAME_ROWS + 1])
        return (
            f"[DataFrame: {n_rows} rows × {header.count(' ') + 1} cols (approx)]\n"
            f"{preview}\n"
            f"... (省略 {max(0, n_rows - MAX_DATAFRAME_ROWS)} 行)"
        )

    def _is_traceback(self, lines: list[str]) -> bool:
        text = "\n".join(lines[:5])
        return "Traceback (most recent call last)" in text

    def _compress_traceback(self, text: str) -> str:
        lines = text.strip().split("\n")
        header = lines[0] if lines else ""
        kept = [header]

        # Keep lines with error location markers (---->)
        for line in lines[1:]:
            if "---->" in line:
                # Keep the marker and the next line (actual code)
                kept.append(line.strip())

        kept.append("... (中间调用链省略) ...")

        # Keep the final error type and message (last meaningful line)
        final_error = ""
        for line in reversed(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith("..."):
                final_error = stripped
                break
        if final_error:
            kept.append(final_error)

        return "\n".join(kept)

    def _truncate_middle(self, text: str) -> str:
        half = MAX_OUTPUT_CHARS // 2
        return text[:half] + f"\n... [{len(text) - MAX_OUTPUT_CHARS} chars omitted] ...\n" + text[-half:]
