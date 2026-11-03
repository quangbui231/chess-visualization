import pandas as pd
import chess.pgn
import io

df = pd.read_csv("output/processed_dataset.csv")
samples = df[df["pgn"].notna()].head(10)

for index, row in samples.iterrows():
    game = chess.pgn.read_game(io.StringIO(str(row["pgn"])))

    if game is None:
        print(f"Dòng {index}: không đọc được ván cờ")
        continue

    plies = sum(1 for _ in game.mainline_moves())
    current = row["total_moves"]

    print(
        f"Dòng {index}: "
        f"total_moves hiện tại = {current}, "
        f"số nước đọc được = {plies}, "
        f"số lượt đầy đủ ước tính = {(plies + 1) // 2}"
    )