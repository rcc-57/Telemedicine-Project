import unittest
from Lab_Work_two import Application, RequestError


class TranscriptTests(unittest.TestCase):
    def setUp(self):
        self.app = Application()
        self.doctor = self.app.dispatch({'method': 'POST', 'path': '/rooms'})[1]
        self.path = '/rooms/' + self.doctor['room']
        self.patient = self.app.dispatch({'method': 'POST', 'path': self.path + '/join'})[1]

    def request(self, who, method='GET', body=None, query=''):
        return self.app.dispatch({'method': method, 'path': self.path + '/transcript' + query,
                                  'token': who['token'], 'body': body or {}})[1]

    def test_only_doctor_can_publish_or_change_status(self):
        epoch = self.request(self.doctor)['epoch']
        for kind in ('status', 'transcript'):
            with self.assertRaises(RequestError) as ctx:
                self.request(self.patient, 'POST', {'kind': kind, 'epoch': epoch, 'text': 'forbidden'})
            self.assertEqual(ctx.exception.status, 403)
        self.request(self.doctor, 'POST', {'kind': 'status', 'epoch': epoch, 'text': 'Работает'})
        row = {'kind': 'transcript', 'epoch': epoch, 'role': 'patient', 'captured': 1000, 'text': '[ИМЯ_1]: боль три дня'}
        self.request(self.doctor, 'POST', row)
        result = self.request(self.patient)
        self.assertEqual(result['status'], 'Работает')
        self.assertEqual(result['rows'][0]['text'], row['text'])
        self.assertEqual(self.request(self.patient, query=f'?after=1&epoch={epoch}')['rows'], [])

    def test_rejoin_clears_history_and_rejects_stale_result(self):
        epoch = self.request(self.doctor)['epoch']
        old = {'kind': 'transcript', 'epoch': epoch, 'role': 'patient', 'captured': 1000, 'text': 'old'}
        self.request(self.doctor, 'POST', old)
        self.app.dispatch({'method': 'POST', 'path': self.path + '/leave', 'token': self.patient['token']})
        self.patient = self.app.dispatch({'method': 'POST', 'path': self.path + '/join'})[1]
        self.assertEqual(self.request(self.patient)['rows'], [])
        with self.assertRaises(RequestError) as ctx:
            self.request(self.doctor, 'POST', old)
        self.assertEqual(ctx.exception.status, 409)

    def test_invalid_and_oversize_rows(self):
        epoch = self.request(self.doctor)['epoch']
        for captured, text in [(float('nan'), 'text'), (0, 'x' * 8001)]:
            with self.assertRaises(RequestError):
                self.request(self.doctor, 'POST', dict(kind='transcript', epoch=epoch, role='patient', captured=captured, text=text))
