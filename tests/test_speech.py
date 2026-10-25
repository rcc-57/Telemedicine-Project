import unittest
from telemed.speech import Anonymizer


class RedactionTests(unittest.TestCase):
    def test_preserves_medical_content_and_stable_names(self):
        cleaner = Anonymizer(lambda text: {'entities': [{'text': 'Иван Петров', 'type': 'PERSON'}]})
        text = 'Иван Петров принимает амоксициллин 500 мг 3 раза в день.'
        self.assertEqual(cleaner.clean(text), '[ИМЯ_1] принимает амоксициллин 500 мг 3 раза в день.')
        self.assertEqual(cleaner.clean('Иван Петров, боль 3 дня'), '[ИМЯ_1], боль 3 дня')

    def test_rules_catch_identifiers_missed_by_model(self):
        cleaner = Anonymizer(lambda text: {'entities': []})
        result = cleaner.clean('Почта test@example.ru, телефон +7 (999) 123-45-67. Доза 500 мг.')
        self.assertNotIn('test@example.ru', result)
        self.assertNotIn('123-45-67', result)
        self.assertIn('500 мг', result)

    def test_overlapping_spans_do_not_leak_suffix(self):
        cleaner = Anonymizer(lambda text: {'entities': [
            {'text': 'Иван', 'type': 'PERSON'}, {'text': 'Иван Петров', 'type': 'PERSON'}]})
        self.assertNotIn('Петров', cleaner.clean('Иван Петров заболел.'))

    def test_invalid_output_fails_closed(self):
        for response in ({}, {'entities': 'none'}, {'entities': [{'text': 'invented', 'type': 'PERSON'}]}):
            with self.subTest(response=response):
                with self.assertRaises(ValueError):
                    Anonymizer(lambda text: response).clean('Секретный текст')

    def test_transport_failure_does_not_return_raw_text(self):
        def fail(text):
            raise TimeoutError()
        with self.assertRaises(TimeoutError):
            Anonymizer(fail).clean('Иван Петров')


class MediaObserverTests(unittest.TestCase):
    def test_muted_speaker_still_observes_once_and_rejects_other_session(self):
        import base64
        from Telemedicine_Client.media import MediaConnection
        observed = []
        connection = MediaConnection.__new__(MediaConnection)
        connection.peer_sid = 'a' * 32
        connection.peer_microphone = True
        connection.speaker = False
        connection.audio_observer = lambda seq, data: observed.append((seq, data))
        connection.observed_audio = set()
        pcm = bytes(640)
        packet = dict(kind='audio', sid='a' * 32, seq=1, data=base64.b64encode(pcm).decode())
        connection._receive(packet)
        connection._receive(packet)
        connection._receive({**packet, 'sid': 'b' * 32, 'seq': 2})
        self.assertEqual(observed, [(1, pcm)])
