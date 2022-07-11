import torch
def profiler(func, *args, **kwargs):
    with torch.autograd.profiler.profile(use_cuda=True, record_shapes=True, with_stack=True) as prof:
        result = func(*args, **kwargs)
    print(prof.key_averages().table(sort_by="self_cuda_time_total"))
    return dict(result=result, prof=prof)

def timer(func, *args, **kwargs):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = func(*args, **kwargs)
    end.record()

    torch.cuda.synchronize()
    print(f"{func.__name__} time: {start.elapsed_time(end)}ms")
    return result


if __name__ == '__main__':
    output = profiler(torch.min, torch.zeros(10000).cuda())
    assert output['result'] == torch.zeros(10000).cuda().min()

    output = timer(torch.min, torch.zeros(10000).cuda())
    assert output == torch.zeros(10000).cuda().min()
