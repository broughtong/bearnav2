import math

def slidingHist(vals, binSize):

    histograms = []

    for offset in range(int(math.floor(-binSize/2)), int(math.floor(binSize/2))):

        histVals = [offset]

        currentCentre = offset
        while currentCentre - (binSize/2.) > min(vals):
            currentCentre -= binSize
            histVals.append(currentCentre)
        currentCentre = offset
        while currentCentre + (binSize/2.) <= max(vals):
            currentCentre += binSize
            histVals.append(currentCentre)

        histVals = sorted(histVals)
        histVals = {i: 0 for i in histVals}

        for val in vals:
            if val < (binSize/2.0) + offset and val >= (-binSize/2.0) + offset:
                histVals[offset] += 1
            elif val < offset:
                currentCentre = offset
                while val < currentCentre - (binSize/2.):
                    currentCentre -= binSize
                histVals[currentCentre] += 1
            elif val > offset:
                currentCentre = offset
                while val >= currentCentre + (binSize/2.):
                    currentCentre += binSize
                histVals[currentCentre] += 1

        histograms.append(histVals)

    return histograms

def hist(vals, binSize):

    histVals = [0.0]
    
    currentCentre = 0.0
    while currentCentre - (binSize/2.) > min(vals):
        currentCentre -= binSize
        histVals.append(currentCentre)
    currentCentre = 0.0
    while currentCentre + (binSize/2.) < max(vals):
        currentCentre += binSize
        histVals.append(currentCentre)
    histVals = sorted(histVals)

    histVals = {i: 0 for i in histVals}

    for val in vals:
        if val < binSize/2.0 and val > -binSize/2.0:
            histVals[0] += 1
        elif val < 0:
            currentCentre = 0.0
            while val < currentCentre - (binSize/2.):
                currentCentre -= binSize
            histVals[currentCentre] += 1
        elif val > 0:
            currentCentre = 0.0
            while val > currentCentre + (binSize/2.):
                currentCentre += binSize
            histVals[currentCentre] += 1

    return histVals

def slidingHistWeighted(vals, binSize, weights):

    histograms = []

    for offset in range(int(math.floor(-binSize/2)), int(math.floor(binSize/2))):

        histVals = [offset]

        currentCentre = offset
        while currentCentre - (binSize/2.) > min(vals):
            currentCentre -= binSize
            histVals.append(currentCentre)
        currentCentre = offset
        while currentCentre + (binSize/2.) <= max(vals):
            currentCentre += binSize
            histVals.append(currentCentre)

        histVals = sorted(histVals)
        histVals = {i: 0 for i in histVals}

        for valIdx in range(len(vals)):
            val = vals[valIdx]
            weight = weights[valIdx]
            if val < (binSize/2.0) + offset and val >= (-binSize/2.0) + offset:
                histVals[offset] += weight
            elif val < offset:
                currentCentre = offset
                while val < currentCentre - (binSize/2.):
                    currentCentre -= binSize
                histVals[currentCentre] += weight
            elif val > offset:
                currentCentre = offset
                while val >= currentCentre + (binSize/2.):
                    currentCentre += binSize
                histVals[currentCentre] += weight

        histograms.append(histVals)

    return histograms

def getHistPeak(hist):

    maxPeak = 0
    maxOffset = 0
    for h in hist:
            for key, value in h.items():
                    if value > maxPeak:
                            maxPeak = value
                            maxOffset = key
    return maxOffset, maxPeak

