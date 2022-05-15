def datacollect(filename=None):##pick the needing data 
    omega=[]
    axim01=[]
    with open(filename,'r') as f:
        for lines in f:
            words=lines.split()
            omega.append(float(words[0]))
            axim01.append(float(words[1]))
    return axim01,omega