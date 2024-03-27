import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from functools import reduce
import numpy as np
from absl import logging
import pickle
from magenta.models.music_vae import TrainedModel
from magenta.models.music_vae import configs
import numpy as np
import tensorflow as tf
import config
import apache_beam as beam
from apache_beam.metrics import Metrics
import note_seq
from note_seq import trim_note_sequence
import song_utils
import copy


class Encode_Song(beam.DoFn):
    """
    Beam DoFn for encoding music sequences into matrices using a pre-trained model.

    This class encapsulates the process of encoding music sequences into matrices using a pre-trained
    music variational autoencoder (VAE) model. It loads the pre-trained model during setup and encodes
    each music sequence passed to it using the `process` method.

    Attributes:
        model_config (dict): Configuration parameters for the pre-trained model.
        model (TrainedModel): Pre-trained music VAE model.
        
    Methods:
        setup(self): Loads the pre-trained model during setup.
        process(self, ns): Processes a single music sequence and yields encoded matrices.

    The `process` method encodes each music sequence into matrices by extracting melodies, encoding
    them using the pre-trained model, and yielding the resulting encoding matrices as pickled objects.
    If a music sequence exceeds a specified duration, it is skipped.
    """
    
    def setup(self):
        """
        Loads the pre-trained model during setup.
        """
        self.cutoff_time = 60 # ns longer than this will be shortened
        self.minimum_time = 60 # minimum length for ns to be processed
        logging.info('Loading pre-trained model')
        self.model_config = config.MUSIC_VAE_CONFIG['melody-2-big']
        self.model = TrainedModel(self.model_config, 
                                  batch_size=1, 
                                  checkpoint_dir_or_path=os.path.expanduser('~/ECE324/cat-mel_2bar_big.tar'))
    
    def process(self, ns):
        """
        Processes a single music sequence and yields encoded matrices.

        Args:
            ns (note_seq.NoteSequence): Music sequence to encode.

        Yields:
            bytes: Pickled representation of the encoding matrices.

        This method processes a single music sequence with length >= 60 and length < 300
        """
        print('Processing %s::%s (%f)' % (ns.id, ns.filename, ns.total_time))
        if ns.total_time > 60 * 5:
            logging.info('Skipping notesequence with >5 minute duration.')
            Metrics.counter('EncodeSong', 'skipped_long_song').inc()
            return
        
        if ns.total_time < self.minimum_time:
            logging.info('Skipping notesequence with <1 minute duration.')
            Metrics.counter('EncodeSong', 'skipped_short_song').inc()
            return
        Metrics.counter('EncodeSong', 'encoding_song').inc()
        
        ns_trim = trim_note_sequence(ns, 0, self.cutoff_time)

        chunk_length = 2
        melodies = song_utils.extract_melodies(ns_trim)
        if not melodies:
            Metrics.counter('EncodeSong', 'extracted_no_melodies').inc()
            return
        Metrics.counter('EncodeSong', 'extracted_melody').inc(len(melodies))
        songs = [
            song_utils.Song(melody, self.model_config.data_converter,
                            chunk_length) for melody in melodies
        ]
        encoding_matrices = song_utils.encode_songs(self.model, songs)

        for matrix in encoding_matrices:
            assert matrix.shape[0] == 3 and matrix.shape[-1] == 512
            if matrix.shape[1] == 0:
                Metrics.counter('EncodeSong', 'skipped_matrix').inc()
                continue
            yield pickle.dumps(matrix)

def encode_ns(input_path, output_path):
    """
    Encode input MIDI files to TFRecord format.

    Args:
        input_path (str): Path to the directory containing note sequences files to encode.
        output_path (str): Path to write the TFRecord file.

    Returns:
        None

    This function encodes MIDI files located at `input_path` into TFRecord format and writes
    the encoded data to a TFRecord file specified by `output_path`.
    """
    print("Starting Pipe")
    with beam.Pipeline() as p:
        p |= 'tfrecord_list' >> beam.Create([os.path.expanduser(input_path)])
        p |= 'read_tfrecord' >> beam.io.tfrecordio.ReadAllFromTFRecord(
            coder=beam.coders.ProtoCoder(note_seq.NoteSequence))
        p |= 'shuffle_input' >> beam.Reshuffle()
        p |= 'encode_song' >> beam.ParDo(Encode_Song())
        p |= 'shuffle_output' >> beam.Reshuffle()
        p |= 'write' >> beam.io.WriteToTFRecord(os.path.expanduser(output_path),
                                                num_shards=1)
    print("Done")

class Mask_Song(beam.DoFn):
    """
    A Beam DoFn for masking music sequences. 
    Songs with length < 60 are skipped, songs with length > 60 are shortened to 60.
    notes between second 15 and 45 are masked
    
    Attributes:
        None

    Methods:
        setup(self): Init values for masking.
        process(self, ns): Masks notes in music sequences longer than 60 seconds.
    """

    def setup(self):
        """
        Initalizing values for masking.
        """
        logging.info('Intializing masking values')
        self.minimum_time = 60 # minimum length for ns to be processed
        self.start_time = 15 # start time of mask
        self.end_time = 45 # end time of mask
    
    def process(self, ns):
        """Masks notes in music sequences longer than 60 seconds.

        Args:
            ns (note_seq.NoteSequence): Music sequence to process.

        Yields:
            note_seq.NoteSequence: Processed music sequence.

        This method masks notes between the 15th and 45th second for sequences
        longer than 60 seconds. Shorter sequences are skipped, longer sequences a shortened to 60 seconds
        """
        print('Masking %s::%s (%f)' % (ns.id, ns.filename, ns.total_time))
        if ns.total_time > 60 * 5:
            logging.info('Skipping notesequence with >5 minute duration.')
            Metrics.counter('EncodeSong', 'skipped_long_song').inc()
            return
        
        if ns.total_time < self.minimum_time:
            logging.info('Skipping %s::%s (%f)', ns.id, ns.filename, ns.total_time)
            Metrics.counter('MaskSong', 'skipped_short_song').inc()
            return  # Skip processing for songs less than 60 seconds
        
        for note in ns.notes:
            if note.start_time >= self.start_time and note.start_time <= self.end_time:
                note.instrument = 0  # Mute the note by setting instrument to 0
        yield ns

def mask_ns(input_path, output_path):
    """
    Mask input MIDI files.

    Args:
        input_path (str): Path to the directory containing note sequences files to encode.
        output_path (str): Path to write the masked note sequence file.

    Returns:
        None

    This function encodes MIDI files located at `input_path` into TFRecord format and writes
    the encoded data to a TFRecord file specified by `output_path`.
    """
    print("Starting Pipe")
    with beam.Pipeline() as p:
        p |= 'tfrecord_list' >> beam.Create([os.path.expanduser(input_path)])
        p |= 'read_tfrecord' >> beam.io.tfrecordio.ReadAllFromTFRecord(
            coder=beam.coders.ProtoCoder(note_seq.NoteSequence))
        p |= 'shuffle_input' >> beam.Reshuffle()
        p |= 'encode_song' >> beam.ParDo(Mask_Song())
        p |= 'shuffle_output' >> beam.Reshuffle()
        p |= 'write' >> beam.io.WriteToTFRecord(os.path.expanduser(output_path), 
                                                num_shards=1,
                                                coder=beam.coders.ProtoCoder(note_seq.NoteSequence))
    print("Done")







    