# %%

def isprime(primes, i):
    for p in primes:
       if i%p == 0:
          return False
    return True

def solution(i):
    primes = []
    soln = ""
    cur_chars = 0
    j=2
    while True:
        if isprime(primes, j):
            primes.append(j)
            cur_chars+=len(str(j))
            if (cur_chars>i):
                soln+=str(j)[-(cur_chars-i):]
                cur_chars=i
            if (len(soln)>=5):
                return soln[:5]
        j+=1
        
            
        
# %%
solution(3)



# %%
