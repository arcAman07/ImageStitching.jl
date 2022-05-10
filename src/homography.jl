struct Homography{T} <: CoordinateTransformations.Transformation
    H::SMatrix{3,3,T,9}
end

function Homography()
    return Homography{Float64}(SMatrix{3,3,Float64}([1 0 0; 0 1 0; 0 0 1]))
end

function Homography{T}() where {T}
    sc = UniformScaling{T}(1)
    m = Matrix(sc, 3, 3)
    return Homography{T}(SMatrix{3,3,T,9}(m))
end

function compute_homography(matches::Array)
    # eigenvector of A^T A with the smallest eigenvalue

    # construct A matrix
    A = zeros(2 * length(matches), 9)
    for (index, match) in enumerate(matches)
        match1, match2 = match
        base_index_x = index * 2 - 1
        base_index_y = 1:3
        A[base_index_x, base_index_y] = float([match1.I...; 1])
        A[base_index_x+1, 4:6] = A[base_index_x, base_index_y]
        A[base_index_x, 7:9] =
            -1.0 * A[base_index_x, base_index_y] * match2.I[1]
        A[base_index_x+1, 7:9] =
            -1.0 * A[base_index_x, base_index_y] * match2.I[2]
    end

    A
    # find the smallest eigenvector, normalize, and reshape
    U, S, Vt = svd(A)

    # normalize the homography at the end, since we know the (3, 3)
    # entry should be 1.
    Homography(reshape(Vt[:, end] ./ Vt[end][end], (3, 3))')
end