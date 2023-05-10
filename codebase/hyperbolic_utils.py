import numpy as np
import torch
from scipy.linalg import block_diag


def moebius_add_mat(A, B, cuda):
    dot_aa = torch.sum(A * A, dim=1)

    dot_bb = torch.sum(B * B, dim=1)
    dot_ab = torch.sum(A * B, dim=1).to(cuda)
    denominator = 1 + 2 * dot_ab + dot_aa * dot_bb
    coef_a = (1 + 2 * dot_ab + dot_bb) / denominator
    coef_b = (1 - dot_aa) / denominator

    return A * coef_a[:, None] + B * coef_b[:, None]


def moebius_mul_mat(A, r):
    norm_v = torch.norm(A, dim=1)
    return A * (torch.tanh(r * torch.arctanh(norm_v)) / (norm_v + 1e-10))[:, None]

def mix_poincare_moebius_add_mat(A, B, num_embs, cuda):
    small_emb_size = int(A.shape[1] / num_embs)
    result = torch.empty_like(A).to(cuda)
    for i in range(num_embs):
        start = i * small_emb_size
        end = (i + 1) * small_emb_size
        result[:, start:end] = moebius_add_mat(A[:, start:end], B[:, start:end], cuda)
    return result


def mix_poincare_moebius_mul_mat(A, r, num_embs, cuda):
    small_emb_size = int(A.shape[1] / num_embs)
    result = torch.empty_like(A).to(cuda)
    for i in range(num_embs):
        start = i * small_emb_size
        end = (i + 1) * small_emb_size
        result[:, start:end] = moebius_mul_mat(A[:, start:end], r)
    return result


# Rotate the rows of X. Each row is split into smaller 2D subspaces. sin_cos_vector contains (sin a, cos a) pairs,
# where a is the angle with which we rotate that 2D portion of a row of X.
def rotate_mat(sin_cos_vector, X, cuda):
    # Create rotation matrix.
    cos_sin_blocks = [[[c, -s], [s, c]] for s, c in sin_cos_vector.detach().cpu().numpy().reshape(-1, 2)]
    rotation_matrix = cos_sin_blocks[0]
    for i in range(1, len(cos_sin_blocks)):
        rotation_matrix = block_diag(rotation_matrix, cos_sin_blocks[i])

    return torch.transpose(
        torch.matmul(torch.from_numpy(rotation_matrix.astype(np.float32)).to(cuda), torch.transpose(X, 0, 1)), 0, 1)


# +
def poincare_distance(vector, other_vectors, cuda):
    diff    = other_vectors.to(cuda) - vector.to(cuda)
    alpha_v = 1.0/ (1 - torch.sum(vector * vector,dim=1))
    beta_w  = 1.0/ (1 - torch.sum(other_vectors * other_vectors,dim=2))
    return torch.acosh(1 + torch.mul(2 * (torch.sum(diff*diff,dim=2)) * alpha_v.to(cuda), beta_w.to(cuda)))

def mix_poincare_distance(x, y, num_embs=50, size=100, cuda=None):
    first  = x.reshape((num_embs, 2))
    second = y.reshape((y.shape[0], num_embs, 2))
    distance = poincare_distance(first, second, cuda)
    return torch.sqrt(torch.sum(distance * distance,dim=1))


# -

def fisher_info_distance(v_i, w_i):
    v_i = v_i.reshape(50, 2)
    w_i = w_i.reshape(50, 2)
    diff = (v_i - w_i)
    multi = np.multiply(diff, diff)
    diff_dot = np.sum(multi, axis=1)

    den = np.multiply(v_i[:, 1], w_i[:, 1])

    half_plane_dists = np.arccosh(1 + diff_dot / (np.multiply(2, den)))
    return np.linalg.norm(half_plane_dists) * np.sqrt(2)


def poincare_ball2half_plane(A, num_embs, cuda):
    small_emb_size = int(A.shape[1] / num_embs)
    result = torch.empty_like(A).to(cuda)
    for i in range(num_embs):
        start = i * small_emb_size
        x = A[:, start].to(cuda)
        y = A[:, start + 1].to(cuda)
        denominator = x * x + (1 - y) * (1 - y)
        result[:, start] = 2 * x / denominator
        result[:, start + 1] = (1 - x * x - y * y) / denominator
    return result

def get_gaussians(embeddings, generic_words, specific_words, aggregate="w", use_wordnet=True, parent_topic_words=None, scaling_factor=1.0, cuda=None):

    if aggregate == "w":
        vectors = embeddings.vectors

    else:
        return None

    try:
        emb = embeddings.num_embs
    except:
        emb = 50

    # Rescale ALL embeddings.
    rescaled_vectors = mix_poincare_moebius_mul_mat(torch.from_numpy(vectors.astype(np.float32)).to(cuda), scaling_factor, emb, cuda)


    if parent_topic_words == None:
        # print('Using poincare-glove paper implementation') > Only generic and specific words of wordnet
        top_and_bottom_levels_avg = torch.mean(rescaled_vectors[torch.cat((generic_words, specific_words)), :],dim=0)
        mean_mat = top_and_bottom_levels_avg.reshape(1, -1).repeat_interleave(rescaled_vectors.shape[0], dim=0).to(cuda)
        recentered_vectors = mix_poincare_moebius_add_mat(-mean_mat, rescaled_vectors, emb, cuda).to(cuda)
        top_levels_avg = torch.mean(recentered_vectors[generic_words, :], dim=0).reshape(-1, 2)
        top_levels_avg_norm = (top_levels_avg / torch.norm(top_levels_avg, dim=1)[:, None]).reshape(-1)
        print(torch.norm(top_levels_avg_norm.reshape(-1, 2), dim=1))
        rotated_vectors = rotate_mat(top_levels_avg_norm.to(cuda), recentered_vectors.to(cuda), cuda)
    else:


        topic_word_index = []
        for word in parent_topic_words:
            if word in embeddings.key_to_index:
                topic_word_index.append(embeddings.key_to_index[word])

        if use_wordnet:
            # Including parent topic words + wordnet words
            topic_word_index = torch.tensor(topic_word_index).to(cuda)
            if (len(topic_word_index)):
                top_and_bottom_levels_avg = torch.mean(rescaled_vectors[torch.cat((generic_words, specific_words)), :], dim=0)
                mean_mat = top_and_bottom_levels_avg.reshape(1, -1).repeat_interleave(rescaled_vectors.shape[0], dim=0).to(cuda)
                recentered_vectors = mix_poincare_moebius_add_mat(-mean_mat, rescaled_vectors, emb, cuda).to(cuda)
                top_levels_avg = torch.mean(recentered_vectors[topic_word_index, :], dim=0).reshape(-1, 2)
                top_levels_avg_norm = ((2.0 * top_levels_avg) / torch.norm(top_levels_avg, dim=1)[:, None]).reshape(-1)
                rotated_vectors = rotate_mat(top_levels_avg_norm, recentered_vectors, cuda)

            else:
                return []

        else:
            # Only parent topic words
            top_and_bottom_levels_avg = torch.mean(rescaled_vectors[topic_word_index, :], dim=0)
            mean_mat = top_and_bottom_levels_avg.reshape(1, -1).repeat_interleave(rescaled_vectors.shape[0], dim=0)
            recentered_vectors = mix_poincare_moebius_add_mat(-mean_mat, rescaled_vectors, emb, cuda)
            top_levels_avg = torch.mean(recentered_vectors[topic_word_index, :], dim=0).reshape(-1, 2)
            top_levels_avg_norm = (top_levels_avg / torch.norm(top_levels_avg, dim=1)[:, None]).reshape(-1)
            rotated_vectors = rotate_mat(top_levels_avg_norm, recentered_vectors, cuda)

    half_plane_vectors = poincare_ball2half_plane(rotated_vectors.to(cuda), emb, cuda)
    gaussians = half_plane_vectors.reshape(-1, emb, 2).to(cuda)
    gaussians[:, :, 0] /= np.sqrt(2)

    return torch.tensor(gaussians).to(cuda)


def get_log_cards_only(index_to_vocab, embeddings, generic_words, specific_words, anchor_word_list=None, use_wordnet=True, cuda=None):
    gaussians = get_gaussians(embeddings, generic_words, specific_words, parent_topic_words=anchor_word_list, use_wordnet=use_wordnet, cuda=cuda)
    
    if (len(gaussians)):
        word_to_card = {}
        for index, word in index_to_vocab.items():
            word_to_card[word] = torch.log(gaussians[index][:, 1]).sum()
        del gaussians
        return word_to_card
    del gaussians
    return {}
