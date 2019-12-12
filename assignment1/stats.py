def calculate_mean(values):
    total = 0
    for i in range(0,len(values)):
        total = total  + values[i]

    mean = total/len(values)
    
    return mean

def calculate_variance_2(values):
    if len(values) == 0:
        return "The list of values you entered was empty"
    else:
        sum_diffsq = 0
        mean = calculate_mean(values)

        for i in range(0,len(values)):
            diff = values[i] - mean
            diffsq = diff**2
            sum_diffsq = sum_diffsq + diffsq

        variance = sum_diffsq/(len(values-1))

        return 'The variance is: ' + str(variance)