import  torch


boundaries = torch.tensor([1, 3, 5, 7, 9])
v = torch.tensor([[0,2,4, 6,8,10]])

ans = torch.bucketize(v, boundaries)
print(ans)
torch.bucketize(v, boundaries, right=True)

print(len(ans[0]))
print(len(boundaries))

print("end")