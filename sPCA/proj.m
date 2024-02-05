function proj = proj(X,Y)
    % Projecting Y onto the tangent space of Stiefel manifold, at X
    proj = Y - X*(X.'*Y+Y.'*X)/2;

end