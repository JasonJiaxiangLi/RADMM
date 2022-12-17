function proj = proj_stiefel(W,Y)

    proj = W - Y*(Y.'*W+W.'*Y)/2;
end