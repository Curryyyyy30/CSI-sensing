function [val,idx] = getperiod(tauVar,winlen)
    ind         = diff(tauVar > max(tauVar)*0.5);
    idxst       = find(ind == 1);
    idxen       = find(ind == -1);
    if length(idxst) ~= length(idxen)
        if length(idxst) > length(idxen)
            idxen = [idxen;length(tauVar)];
        else
            idxst = [1;idxst];
        end
    end
    idx         = [idxst, idxen];
    Oneway_idx  = reshape(idx.',[],1).';
    diff_idx    = diff(Oneway_idx);
    inter       = diff_idx(1:2:end);
    inter_error = inter < winlen/4;
    idx(inter_error,:)      = [];
    Oneway_idx  = reshape(idx.',[],1).';
    diff_idx    = diff(Oneway_idx);
    cross       = diff_idx(2:2:end);
    cross_error = find(cross < winlen/10)/2;
    cross_error = fliplr(cross_error);
    for i = 1 : length(cross_error)
        idx(cross_error(i),2)   = idx(cross_error(i)+1,2);
        idx(cross_error(i)+1,:) = [];
    end
    val         = tauVar(idx);
end