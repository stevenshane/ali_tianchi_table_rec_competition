"""主程序：python main.py [--ids 1,5,10-20] [--stage all|recognize|answer] [--force] [--workers 4]"""
import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from table_rec import pipeline, submission
from table_rec.qa import answerer, tasks as task_mod


def setup_logging():
    config.LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(config.LOG_DIR / "run.log", encoding="utf-8")],
    )
    for noisy in ("httpx", "httpx2", "anthropic", "pdfminer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main():
    ap = argparse.ArgumentParser(description="复杂表格识别与内容理解")
    ap.add_argument("--ids", help="题目 id，逗号分隔，支持区间，如 1,5,10-20；缺省为全部")
    ap.add_argument("--stage", choices=["all", "recognize", "answer"], default="all")
    ap.add_argument("--force", action="store_true", help="忽略 files_json 缓存重新识别")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    setup_logging()
    log = logging.getLogger("main")

    ids = task_mod.parse_ids(args.ids)
    todo = task_mod.load_tasks(ids)
    if not todo:
        log.error("没有匹配的题目: %s", args.ids)
        return 1
    files = sorted({t["file_name"] for t in todo})
    log.info("题目 %d 道，涉及文件 %d 个", len(todo), len(files))

    if args.stage in ("all", "recognize"):
        with ThreadPoolExecutor(args.workers) as ex:
            futs = {ex.submit(pipeline.recognize_file, config.FILES_DIR / f, args.force): f for f in files}
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception:
                    log.exception("识别失败: %s", futs[fut])

    if args.stage in ("all", "answer"):
        results = []
        with ThreadPoolExecutor(args.workers) as ex:
            for res in ex.map(answerer.answer_task, todo):
                results.append(res)
        for r in results:
            log.info("题 %s -> %s%s", r["id"], r["answer"][:120],
                     f"  [规则提示: {r['issues']}]" if r["issues"] else "")
        all_ids = [t["id"] for t in task_mod.load_tasks()]
        out = submission.write(results, all_ids)
        (config.LOG_DIR / "answers.jsonl").open("a", encoding="utf-8").writelines(
            json.dumps(r, ensure_ascii=False) + "\n" for r in results)
        log.info("已写出 %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
