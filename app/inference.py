import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def generate_caption(model, tokenizer, features, max_length=34, beam_index=3):
    start = [tokenizer.word_index['startseq']]  # Start with the start token

    def remove_endseq(caption):
        """Helper function to remove redundant 'endseq' tokens."""
        caption = [word for word in caption if word != tokenizer.word_index['endseq']]
        return caption

    if beam_index == 1:
        # Greedy search
        caption = start
        for _ in range(max_length):
            seq = pad_sequences([caption], maxlen=max_length, padding='post')
            preds = model.predict([features, seq], verbose=0)
            word_id = np.argmax(preds[0])
            # Stop if the 'endseq' token is generated or if the prediction is empty
            if word_id == 0 or word_id == tokenizer.word_index['endseq']:
                break
            caption.append(word_id)

        # Clean up the caption by removing 'startseq' and redundant 'endseq'
        caption = remove_endseq(caption[1:])
        return ' '.join(tokenizer.index_word.get(i, '') for i in caption)

    else:
        # Beam search
        sequences = [[start, 0.0]]  # Initial sequence with a score of 0
        for _ in range(max_length):
            all_candidates = []
            for seq, score in sequences:
                padded = pad_sequences([seq], maxlen=max_length, padding='post')
                preds = model.predict([features, padded], verbose=0)
                top_preds = np.argsort(preds[0])[-beam_index:]  # Get the top beam_index predictions

                for word_id in top_preds:
                    new_seq = seq + [word_id]
                    new_score = score - np.log(preds[0][word_id] + 1e-10)
                    all_candidates.append([new_seq, new_score])

            # Sort all candidates based on score and select the best
            sequences = sorted(all_candidates, key=lambda tup: tup[1])[:beam_index]

        # Extract the best sequence from the beam search
        final_seq = sequences[0][0]
        final_seq = remove_endseq(final_seq[1:])  # Clean up by removing redundant 'endseq'
        return ' '.join(tokenizer.index_word.get(i, '') for i in final_seq)
