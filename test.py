import sys

input = sys.stdin.readline

N, K = map(int, input().split())

Share_Card = list(map(int, input().split()))
Team_Card = list(map(int, input().split()))

Case = []
for S in Share_Card:
    for T in Team_Card:
       Case.append((S*T, S, T))

Case.sort()

print(Case) 