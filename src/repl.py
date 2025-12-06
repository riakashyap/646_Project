"""
Copyright:

  Copyright © 2025 bdunahu

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:

  This file includes a small interactive REPL.

Code:
"""

import code
import threading
import time
import sys

class Repl(code.InteractiveConsole):
    def __init__(self, corag):
        super().__init__()
        self.corag = corag
        self.verdict_keys = ["false", "true", "inconclusive"]
        self.loading_thread = None
        self.loading = False

    def obnoxious_load(self):
        while self.loading:
            for f in ["|", "/", "-", "\\"]:
                print(f"\rThinking...{f}", end="", flush=True)
                time.sleep(0.2)

    def runsource(self, source, filename="<input>", symbol="single"):
        if source.strip().lower() in ["quit", "exit", "bye"]:
            return False
        out = True
        if len(source) > 0:
            out = self.corag.run(source)
        return out

    def respond(self, out):
        claim = out["claim"]
        verdict = out["verdict_raw"]
        verdict_bool = out["verdict"]
        print(f"\nWe considered the claim: {claim}")
        print(f"\nThe questions we posed and answered were:")
        for e in out["qa_pairs"]:
            print(f"\tQuestion: {e[0]}\n\t\t{e[1]}")
        print()
        print(f"Based on this, we logically conclude:\n\t{verdict}")
        print(f"Our final answer: {self.verdict_keys[verdict_bool]}\n\n")

    def interact(self, banner=None, exitmsg=None):
        if banner:
            print(banner)
        while True:
            try:
                source = input('Say: ')
                self.loading = True
                self.loading_thread = threading.Thread(target=self.obnoxious_load)
                self.loading_thread.start()

                out = self.runsource(source)

                self.loading = False
                self.loading_thread.join()

                if not out:
                    break
                if isinstance(out, dict):
                    self.respond(out)
            except EOFError:
                print(exitmsg)
                break
            except KeyboardInterrupt:
                print(exitmsg)
                break
