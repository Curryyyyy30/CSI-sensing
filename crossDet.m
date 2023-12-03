function [shotstart , shotend , peakind] = crossDet(tauEst)
    
    winlen  = 70;
    tauEst  = wden(tauEst,'heursure','s','one',2,'sym2').';                 %小波去噪
    figure('color','white')
    subplot(2,1,1)
    plot(tauEst)
    tauVar  = movvar(tauEst,winlen);
    tauVar  = medfilt1(tauVar,6);
    subplot(2,1,2)
    plot(tauVar)
    [val,idx]   = getperiod(tauVar,winlen);
    if isempty(idx)
        shotstart   =[];
        shotend     =[];
        peakind     =[];
    else
        shotstart   = idx(:,1);
        shotend     = idx(:,2);
        for i = 1 : length(shotstart)
            peakind(i) = find(tauEst == max(tauEst(shotstart(i):shotend(i))));
        end
        if max(val) < 10e-20
            shotstart   =[];
            shotend     =[];
            peakind     =[];
        end
    end
    close all
end