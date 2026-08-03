import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.test.ci.ci_register import register_amd_ci


register_amd_ci(est_time=40, suite="stage-b-test-1-gpu-large-amd")


class TestAiterGDNBackendRegistration(unittest.TestCase):
    def test_aiter_backend_is_registered(self):
        self.assertEqual(LinearAttnKernelBackend.AITER.value, "aiter")

    def test_decode_backend_selection(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_aiter import (
            select_aiter_gdn_decode_backend,
        )

        self.assertEqual(
            select_aiter_gdn_decode_backend(
                2, 8, hip_available=True, fly_available=True
            ),
            "hip",
        )
        self.assertEqual(
            select_aiter_gdn_decode_backend(
                8, 24, hip_available=True, fly_available=True
            ),
            "flydsl",
        )
        self.assertEqual(
            select_aiter_gdn_decode_backend(
                8, 24, hip_available=False, fly_available=False
            ),
            "triton",
        )

    def test_hip_decode_runtime_guard(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_aiter import (
            supports_hip_gdn_decode_runtime,
        )

        common = dict(
            local_num_k_heads=2,
            local_num_v_heads=8,
            q_dtype=torch.bfloat16,
            k_dtype=torch.bfloat16,
            v_dtype=torch.bfloat16,
            a_dtype=torch.bfloat16,
            b_dtype=torch.bfloat16,
            dt_bias_dtype=torch.bfloat16,
            state_dtype=torch.float32,
            head_k_dim=128,
            head_v_dim=128,
            state_shape=(64, 8, 128, 128),
        )
        self.assertTrue(supports_hip_gdn_decode_runtime(**common))
        self.assertFalse(
            supports_hip_gdn_decode_runtime(**{**common, "state_dtype": torch.bfloat16})
        )
        self.assertFalse(
            supports_hip_gdn_decode_runtime(
                **{**common, "local_num_v_heads": 24, "state_shape": (64, 24, 128, 128)}
            )
        )

    def test_decode_uses_hip_then_flydsl_then_triton(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_aiter import (
            AiterGDNKernel,
        )

        fallback = mock.Mock()
        fallback.decode.return_value = "triton"
        hip_decode = mock.Mock(return_value="hip")
        fly_decode = mock.Mock(return_value="flydsl")
        kernel = AiterGDNKernel(
            fallback_kernel=fallback,
            hip_decode=hip_decode,
            fly_decode=fly_decode,
            hip_arch_supported=True,
        )

        def inputs(k_heads, v_heads, state_dtype=torch.float32):
            return dict(
                q=torch.empty(1, 2, k_heads, 128, dtype=torch.bfloat16),
                k=torch.empty(1, 2, k_heads, 128, dtype=torch.bfloat16),
                v=torch.empty(1, 2, v_heads, 128, dtype=torch.bfloat16),
                a=torch.empty(1, 2, v_heads, dtype=torch.bfloat16),
                b=torch.empty(1, 2, v_heads, dtype=torch.bfloat16),
                A_log=torch.empty(v_heads, dtype=torch.float32),
                dt_bias=torch.empty(v_heads, dtype=torch.bfloat16),
                ssm_states=torch.empty(8, v_heads, 128, 128, dtype=state_dtype),
                cache_indices=torch.tensor([0, 1], dtype=torch.int32),
                query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
            )

        self.assertEqual(kernel.decode(**inputs(2, 8)), "hip")
        self.assertEqual(kernel.decode(**inputs(8, 24)), "flydsl")

        fallback_only = AiterGDNKernel(
            fallback_kernel=fallback,
            hip_decode=hip_decode,
            fly_decode=None,
            hip_arch_supported=True,
        )
        self.assertEqual(
            fallback_only.decode(**inputs(2, 8, state_dtype=torch.bfloat16)),
            "triton",
        )
        self.assertEqual(
            fallback_only.decode(**inputs(2, 8), replayssm_d=torch.empty(1)),
            "triton",
        )

    def test_decode_sort_cache_reset_is_forwarded(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_aiter import (
            AiterGDNKernel,
        )

        reset = mock.Mock()
        kernel = AiterGDNKernel(
            fallback_kernel=mock.Mock(),
            hip_decode=mock.Mock(),
            fly_decode=None,
            reset_sort_cache=reset,
        )
        kernel.reset_decode_cache()
        reset.assert_called_once_with()

    def test_decode_ignores_cuda_graph_padding(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_aiter import (
            AiterGDNKernel,
        )

        fallback = mock.Mock()

        def hip_decode(**kwargs):
            return torch.ones_like(kwargs["v"])

        kernel = AiterGDNKernel(
            fallback_kernel=fallback,
            hip_decode=hip_decode,
            fly_decode=None,
            hip_arch_supported=True,
        )
        output = kernel.decode(
            q=torch.empty(1, 2, 2, 128, dtype=torch.bfloat16),
            k=torch.empty(1, 2, 2, 128, dtype=torch.bfloat16),
            v=torch.empty(1, 2, 8, 128, dtype=torch.bfloat16),
            a=torch.empty(1, 2, 8, dtype=torch.bfloat16),
            b=torch.empty(1, 2, 8, dtype=torch.bfloat16),
            A_log=torch.empty(8, dtype=torch.float32),
            dt_bias=torch.empty(8, dtype=torch.bfloat16),
            ssm_states=torch.empty(4, 8, 128, 128, dtype=torch.float32),
            cache_indices=torch.tensor([1, -1], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 1, 1], dtype=torch.int32),
            active_batch_size=1,
        )
        self.assertEqual(output.shape, (1, 2, 8, 128))
        torch.testing.assert_close(output[:, :1], torch.ones_like(output[:, :1]))
        torch.testing.assert_close(output[:, 1:], torch.zeros_like(output[:, 1:]))
        fallback.decode.assert_not_called()

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.version.hip is not None,
        "ROCm is required",
    )
    def test_decode_uses_triton_during_cuda_graph_capture(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_aiter import (
            AiterGDNKernel,
        )

        fallback = mock.Mock()
        fallback.decode.return_value = "triton-graph"
        hip_decode = mock.Mock()
        kernel = AiterGDNKernel(
            fallback_kernel=fallback,
            hip_decode=hip_decode,
            fly_decode=None,
            hip_arch_supported=True,
        )
        with mock.patch("torch.cuda.is_current_stream_capturing", return_value=True):
            output = kernel.decode(
                q=torch.empty(1, 2, 2, 128, device="cuda", dtype=torch.bfloat16),
                k=torch.empty(1, 2, 2, 128, device="cuda", dtype=torch.bfloat16),
                v=torch.empty(1, 2, 8, 128, device="cuda", dtype=torch.bfloat16),
                a=torch.empty(1, 2, 8, device="cuda", dtype=torch.bfloat16),
                b=torch.empty(1, 2, 8, device="cuda", dtype=torch.bfloat16),
                A_log=torch.empty(8, device="cuda", dtype=torch.float32),
                dt_bias=torch.empty(8, device="cuda", dtype=torch.bfloat16),
                ssm_states=torch.empty(
                    4, 8, 128, 128, device="cuda", dtype=torch.float32
                ),
                cache_indices=torch.tensor([1, 2], device="cuda", dtype=torch.int32),
                query_start_loc=torch.tensor(
                    [0, 1, 2], device="cuda", dtype=torch.int32
                ),
            )
        self.assertEqual(output, "triton-graph")
        fallback.decode.assert_called_once()
        hip_decode.assert_not_called()

    def test_dispatcher_can_select_aiter_per_mode(self):
        from sglang.srt.layers.attention.linear.gdn_backend import (
            GDNKernelDispatcher,
        )
        from sglang.srt.layers.attention.linear.kernels.gdn_aiter import (
            AiterGDNKernel,
        )
        from sglang.srt.layers.attention.linear.kernels.gdn_triton import (
            TritonGDNKernel,
        )

        with mock.patch.object(
            AiterGDNKernel,
            "__init__",
            lambda self, fallback_kernel=None: setattr(
                self, "fallback_kernel", fallback_kernel
            ),
        ):
            dispatcher = GDNKernelDispatcher(
                LinearAttnKernelBackend.AITER,
                LinearAttnKernelBackend.AITER,
            )

        self.assertIsInstance(dispatcher.decode_kernel, AiterGDNKernel)
        self.assertIsInstance(dispatcher.extend_kernel, AiterGDNKernel)
        self.assertIsInstance(dispatcher.verify_kernel, TritonGDNKernel)
        self.assertFalse(dispatcher.supports_packed_decode)

    def test_dispatcher_forwards_decode_cache_reset(self):
        from sglang.srt.layers.attention.linear.gdn_backend import (
            GDNKernelDispatcher,
        )

        dispatcher = object.__new__(GDNKernelDispatcher)
        dispatcher.decode_kernel = mock.Mock()
        dispatcher.reset_decode_cache()
        dispatcher.decode_kernel.reset_decode_cache.assert_called_once_with()

    def test_backend_prepares_active_decode_batch_for_graph_padding(self):
        from sglang.srt.layers.attention.linear.gdn_backend import (
            GDNAttnBackend,
            _forward_batch_has_padding,
        )

        backend = object.__new__(GDNAttnBackend)
        backend.kernel_dispatcher = mock.Mock()
        forward_mode = mock.Mock()
        forward_mode.is_decode_or_idle.return_value = True
        forward_batch = SimpleNamespace(
            forward_mode=forward_mode,
            batch_size=8,
            num_padding=3,
        )
        backend._prepare_aiter_forward_metadata(forward_batch)
        self.assertEqual(backend._aiter_decode_active_batch_size, 5)
        backend.kernel_dispatcher.reset_decode_cache.assert_called_once_with()
        self.assertTrue(
            _forward_batch_has_padding(
                SimpleNamespace(
                    batch_size=8,
                    num_padding=0,
                    _original_batch_size=5,
                )
            )
        )

    def test_prefill_uses_aiter_and_falls_back_when_intermediate_h_is_required(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_aiter import (
            AiterGDNKernel,
        )

        fallback = mock.Mock()
        fallback.extend.return_value = ("triton", None, "h")

        def prefill(**kwargs):
            self.assertTrue(kwargs["q"].is_contiguous())
            self.assertTrue(kwargs["k"].is_contiguous())
            self.assertTrue(kwargs["v"].is_contiguous())
            self.assertTrue(kwargs["g"].is_contiguous())
            self.assertTrue(kwargs["beta"].is_contiguous())
            return "aiter", "full-pool-state"

        prefill = mock.Mock(side_effect=prefill)
        kernel = AiterGDNKernel(
            fallback_kernel=fallback,
            hip_decode=None,
            fly_decode=None,
            prefill_vk=prefill,
            prefill_intermediate_ops=None,
        )
        kwargs = dict(
            q=torch.empty(1, 5, 2, 256, dtype=torch.bfloat16)[..., ::2],
            k=torch.empty(1, 5, 2, 256, dtype=torch.bfloat16)[..., ::2],
            v=torch.empty(1, 5, 8, 256, dtype=torch.bfloat16)[..., ::2],
            g=torch.empty(1, 5, 16, dtype=torch.float32)[..., ::2],
            beta=torch.empty(1, 5, 16, dtype=torch.float32)[..., ::2],
            ssm_states=torch.empty(4, 8, 128, 128, dtype=torch.float32),
            cache_indices=torch.tensor([1], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 5], dtype=torch.int32),
        )

        output, final_state, h = kernel.extend(**kwargs, return_intermediate_h=False)
        self.assertEqual(output, "aiter")
        self.assertIsNone(final_state)
        self.assertIsNone(h)
        prefill.assert_called_once()

        self.assertEqual(
            kernel.extend(**kwargs, return_intermediate_h=True),
            ("triton", None, "h"),
        )
        fallback.extend.assert_called_once()

    def test_prefill_returns_aiter_intermediate_h_when_ops_are_available(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_aiter import (
            AiterGDNKernel,
        )

        fallback = mock.Mock()
        h_expected = torch.empty(1, 2, 8, 128, 128)
        ops = {
            "cumsum": mock.Mock(return_value=(torch.empty(1), torch.empty(1))),
            "solve": mock.Mock(return_value=(torch.empty(1), torch.empty(1))),
            "chunk_h": mock.Mock(return_value=(h_expected, torch.empty(1), None)),
            "chunk_o": mock.Mock(return_value="aiter-h"),
        }
        kernel = AiterGDNKernel(
            fallback_kernel=fallback,
            hip_decode=None,
            fly_decode=None,
            prefill_vk=mock.Mock(),
            prefill_intermediate_ops=ops,
            l2norm=lambda tensor: tensor,
        )
        output, final_state, h = kernel.extend(
            q=torch.empty(1, 5, 2, 128, dtype=torch.bfloat16),
            k=torch.empty(1, 5, 2, 128, dtype=torch.bfloat16),
            v=torch.empty(1, 5, 8, 128, dtype=torch.bfloat16),
            g=torch.empty(1, 5, 8, dtype=torch.float32),
            beta=torch.empty(1, 5, 8, dtype=torch.float32),
            ssm_states=torch.empty(4, 8, 128, 128, dtype=torch.float32),
            cache_indices=torch.tensor([1], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 5], dtype=torch.int32),
            return_intermediate_h=True,
        )
        self.assertEqual(output, "aiter-h")
        self.assertIsNone(final_state)
        self.assertIs(h, h_expected)
        fallback.extend.assert_not_called()

    def test_final_state_tracking_does_not_require_intermediate_h(self):
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            MambaAttnBackendBase,
        )

        backend = object.__new__(MambaAttnBackendBase)
        states = torch.arange(24, dtype=torch.float32).view(6, 4)
        metadata = SimpleNamespace(
            has_mamba_track_mask=True,
            track_ssm_h_src=torch.empty(0, dtype=torch.int64),
            track_ssm_h_dst=torch.empty(0, dtype=torch.int64),
            track_ssm_final_src=torch.tensor([1], dtype=torch.int64),
            track_ssm_final_dst=torch.tensor([4], dtype=torch.int64),
        )
        expected = states[1].clone()

        backend._track_mamba_state_extend(
            SimpleNamespace(),
            None,
            states,
            metadata,
        )
        torch.testing.assert_close(states[4], expected)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.version.hip is not None,
        "ROCm is required",
    )
    def test_aiter_decode_matches_triton_for_397b_tp8(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_aiter import (
            AiterGDNKernel,
        )
        from sglang.srt.layers.attention.linear.kernels.gdn_triton import (
            TritonGDNKernel,
        )

        torch.manual_seed(7)
        batch, k_heads, v_heads, dim = 2, 2, 8, 128
        q = torch.randn(1, batch, k_heads, dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn(1, batch, v_heads, dim, device="cuda", dtype=torch.bfloat16)
        a = torch.randn(1, batch, v_heads, device="cuda", dtype=torch.bfloat16)
        b = torch.randn_like(a)
        A_log = torch.randn(v_heads, device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(v_heads, device="cuda", dtype=torch.bfloat16)
        indices = torch.tensor([1, 4], device="cuda", dtype=torch.int32)
        starts = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
        state = torch.randn(6, v_heads, dim, dim, device="cuda", dtype=torch.float32)
        state_ref = state.clone()
        state_actual = state.clone()

        triton = TritonGDNKernel()
        output_ref = triton.decode(
            q,
            k,
            v,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=state_ref,
            cache_indices=indices,
            query_start_loc=starts,
        )
        output = AiterGDNKernel(fallback_kernel=triton).decode(
            q,
            k,
            v,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=state_actual,
            cache_indices=indices,
            query_start_loc=starts,
        )
        torch.testing.assert_close(output, output_ref, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(state_actual, state_ref, rtol=1e-3, atol=1e-3)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.version.hip is not None,
        "ROCm is required",
    )
    def test_aiter_prefill_matches_triton_for_397b_tp8(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_aiter import (
            AiterGDNKernel,
        )
        from sglang.srt.layers.attention.linear.kernels.gdn_triton import (
            TritonGDNKernel,
        )

        torch.manual_seed(11)
        tokens, k_heads, v_heads, dim = 65, 2, 8, 128
        q = torch.randn(1, tokens, k_heads, dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn(1, tokens, v_heads, dim, device="cuda", dtype=torch.bfloat16)
        g = -torch.nn.functional.softplus(
            torch.randn(1, tokens, v_heads, device="cuda", dtype=torch.float32)
        )
        beta = torch.sigmoid(torch.randn_like(g))
        indices = torch.tensor([1, 4], device="cuda", dtype=torch.int32)
        starts = torch.tensor([0, 31, 65], device="cuda", dtype=torch.int32)
        state = torch.randn(6, v_heads, dim, dim, device="cuda", dtype=torch.float32)
        state_ref = state.clone()
        state_actual = state.clone()

        triton = TritonGDNKernel()
        output_ref, _, _ = triton.extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=state_ref,
            cache_indices=indices,
            query_start_loc=starts,
        )
        with mock.patch.dict("os.environ", {"SGLANG_GDN_PREFILL_OPT_VK_K5": "triton"}):
            output, _, h = AiterGDNKernel(
                fallback_kernel=triton,
                hip_decode=None,
                fly_decode=None,
            ).extend(
                q,
                k,
                v,
                g,
                beta,
                ssm_states=state_actual,
                cache_indices=indices,
                query_start_loc=starts,
                return_intermediate_h=False,
            )
        self.assertIsNone(h)
        torch.testing.assert_close(output, output_ref, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(state_actual, state_ref, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
