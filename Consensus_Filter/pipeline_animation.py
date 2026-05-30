from manim import *


class ConsensusFilterScene(ThreeDScene):
    def construct(self):
        query_text = "Tell me the closest place to buy a tasty deep-dish pizza"
        provider_specs = [
            {
                "name": "Gemini",
                "color": TEAL_D,
                "response": [
                    "Joey's Pizza | 0.3 mi | $ | 4.7",
                    "Chicken Spot | 0.2 mi | $$ | 4.1",
                ],
                "vectors": [[0.12, 0.08, 0.0], [0.32, 0.0, 0.18]],
            },
            {
                "name": "Ollama",
                "color": GOLD_D,
                "response": [
                    "Joey's Pizza | 0.3 mi | $ | 4.6",
                    "Windy Slice | 0.4 mi | $$ | 4.5",
                ],
                "vectors": [[0.12, 0.08, 0.02], [0.36, 0.14, 0.06]],
            },
            {
                "name": "Consensus",
                "color": BLUE_D,
                "response": [
                    "Joey's Pizza survives overlap",
                    "Other candidates drift further out",
                ],
                "vectors": [[0.12, 0.08, 0.01], [0.36, 0.14, 0.06]],
            },
        ]

        title = Text("Consensus Filter Pipeline", font_size=36, weight=BOLD)
        title.to_edge(UP)

        query_box = RoundedRectangle(corner_radius=0.2, width=6.4, height=1.1, color=WHITE)
        query_box.set_fill(BLACK, opacity=0.9)
        query_label = Text("User Query", font_size=26, weight=BOLD)
        query_value = Text(query_text, font_size=20)
        query_value.scale_to_fit_width(query_box.width - 0.45)
        query_group = VGroup(query_box, query_label, query_value.arrange if False else query_value)
        query_label.move_to(query_box.get_center() + UP * 0.2)
        query_value.move_to(query_box.get_center() + DOWN * 0.22)
        query_group = VGroup(query_box, query_label, query_value)
        query_group.move_to(UP * 2.2)

        self.play(FadeIn(title, shift=UP * 0.3), FadeIn(query_group, shift=UP * 0.4))
        self.wait(0.4)

        provider_group = VGroup()
        response_groups = []
        vector_groups = []
        connectors = []

        provider_x_positions = [-4.6, 0.0, 4.6]
        for index, spec in enumerate(provider_specs):
            provider_box = RoundedRectangle(
                corner_radius=0.2,
                width=2.6,
                height=0.9,
                color=spec["color"],
            )
            provider_box.set_fill(spec["color"], opacity=0.15)
            provider_label = Text(spec["name"], font_size=24, weight=BOLD, color=spec["color"])
            provider = VGroup(provider_box, provider_label)
            provider_label.move_to(provider_box.get_center())
            provider.move_to([provider_x_positions[index], 0.95, 0])
            provider_group.add(provider)

            query_arrow = Arrow(
                query_group.get_bottom() + DOWN * 0.05,
                provider.get_top() + UP * 0.05,
                buff=0.12,
                stroke_width=5,
                color=spec["color"],
            )
            connectors.append(query_arrow)

            response_box = RoundedRectangle(
                corner_radius=0.18,
                width=3.1,
                height=1.45,
                color=GREY_B,
            )
            response_box.set_fill(GREY_E, opacity=0.92)
            response_lines = VGroup(*[
                Text(line, font_size=16, color=WHITE) for line in spec["response"]
            ]).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
            response_lines.scale_to_fit_width(response_box.width - 0.35)
            response_group = VGroup(response_box, response_lines)
            response_lines.move_to(response_box.get_center())
            response_group.move_to([provider_x_positions[index], -0.75, 0])
            response_groups.append(response_group)

            vector_box = RoundedRectangle(
                corner_radius=0.18,
                width=2.7,
                height=1.25,
                color=spec["color"],
            )
            vector_box.set_fill(spec["color"], opacity=0.1)
            vector_title = Text("Vectorized", font_size=18, weight=BOLD)
            vector_lines = VGroup(*[
                Text(str(vector), font_size=16) for vector in spec["vectors"]
            ]).arrange(DOWN, buff=0.1)
            vector_lines.scale_to_fit_width(vector_box.width - 0.35)
            vector_group = VGroup(vector_box, vector_title, vector_lines)
            vector_title.move_to(vector_box.get_center() + UP * 0.36)
            vector_lines.move_to(vector_box.get_center() + DOWN * 0.16)
            vector_group.move_to([provider_x_positions[index], -2.45, 0])
            vector_groups.append(vector_group)

        for provider, response_group, vector_group, arrow in zip(
            provider_group, response_groups, vector_groups, connectors
        ):
            down_arrow = Arrow(
                provider.get_bottom() + DOWN * 0.06,
                response_group.get_top() + UP * 0.06,
                buff=0.12,
                stroke_width=4,
                color=GREY_B,
            )
            vector_arrow = Arrow(
                response_group.get_bottom() + DOWN * 0.06,
                vector_group.get_top() + UP * 0.06,
                buff=0.12,
                stroke_width=4,
                color=GREY_B,
            )

            self.play(Create(arrow), FadeIn(provider, shift=DOWN * 0.2), run_time=0.7)
            self.play(Create(down_arrow), FadeIn(response_group, shift=DOWN * 0.2), run_time=0.7)
            self.play(Create(vector_arrow), FadeIn(vector_group, shift=DOWN * 0.2), run_time=0.7)
            self.wait(0.2)

        query_vector_box = RoundedRectangle(corner_radius=0.18, width=3.7, height=1.0, color=WHITE)
        query_vector_box.set_fill(BLACK, opacity=0.88)
        query_vector_title = Text("Query -> feature vector", font_size=22, weight=BOLD)
        query_vector_value = Text("[price gap, distance gap, rating gap]", font_size=18)
        query_vector_title.scale_to_fit_width(query_vector_box.width - 0.35)
        query_vector_value.scale_to_fit_width(query_vector_box.width - 0.35)
        query_vector_group = VGroup(query_vector_box, query_vector_title, query_vector_value)
        query_vector_title.move_to(query_vector_box.get_center() + UP * 0.18)
        query_vector_value.move_to(query_vector_box.get_center() + DOWN * 0.18)
        query_vector_group.move_to(LEFT * 3.8 + DOWN * 3.6)

        origin_box = RoundedRectangle(corner_radius=0.18, width=2.4, height=0.95, color=RED_C)
        origin_box.set_fill(RED_D, opacity=0.14)
        origin_text = Text("Normalize to [0, 0, 0]", font_size=21, weight=BOLD, color=RED_C)
        origin_text.scale_to_fit_width(origin_box.width - 0.3)
        origin_group = VGroup(origin_box, origin_text)
        origin_text.move_to(origin_box.get_center())
        origin_group.move_to(RIGHT * 3.8 + DOWN * 3.6)

        query_to_vector = CurvedArrow(
            query_group.get_bottom() + LEFT * 0.4,
            query_vector_group.get_top() + UP * 0.06,
            angle=0.45,
            color=WHITE,
            stroke_width=4,
        )
        vector_to_origin = Arrow(
            query_vector_group.get_right() + RIGHT * 0.08,
            origin_group.get_left() + LEFT * 0.08,
            buff=0.12,
            stroke_width=5,
            color=RED_C,
        )

        self.play(Create(query_to_vector), FadeIn(query_vector_group, shift=LEFT * 0.2))
        self.play(Create(vector_to_origin), FadeIn(origin_group, shift=RIGHT * 0.2))
        self.wait(0.6)

        pipeline_mobjects = VGroup(
            title,
            query_group,
            provider_group,
            *response_groups,
            *vector_groups,
            query_vector_group,
            origin_group,
            *connectors,
            query_to_vector,
            vector_to_origin,
        )

        self.play(FadeOut(pipeline_mobjects, shift=UP * 0.25), run_time=0.9)

        axes = ThreeDAxes(
            x_range=[0, 0.45, 0.1],
            y_range=[0, 0.25, 0.05],
            z_range=[0, 0.2, 0.05],
            x_length=5.4,
            y_length=4.2,
            z_length=3.8,
        )
        axes_labels = VGroup(
            axes.get_x_axis_label(Text("Price Gap", font_size=22)),
            axes.get_y_axis_label(Text("Distance Gap", font_size=22)),
            axes.get_z_axis_label(Text("Rating Gap", font_size=22)),
        )
        axes_group = VGroup(axes, axes_labels)
        axes_group.scale(0.98)
        axes_group.move_to(DOWN * 0.1)

        self.play(FadeIn(axes_group, shift=UP * 0.3))
        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES, zoom=0.95)

        origin_dot = Dot3D(point=axes.c2p(0, 0, 0), radius=0.08, color=WHITE)
        origin_label = Text("Query Origin [0, 0, 0]", font_size=20)
        origin_label.next_to(axes_group, DOWN, buff=0.3).shift(RIGHT * 1.1)

        self.play(FadeIn(origin_dot), FadeIn(origin_label, shift=UP * 0.2))

        plot_points = [
            {
                "coords": (0.12, 0.08, 0.0),
                "label": "Gemini: Joey's",
                "color": TEAL_D,
                "winner": False,
            },
            {
                "coords": (0.32, 0.0, 0.18),
                "label": "Gemini: Chicken Spot",
                "color": TEAL_D,
                "winner": False,
            },
            {
                "coords": (0.12, 0.08, 0.02),
                "label": "Ollama: Joey's",
                "color": GOLD_D,
                "winner": False,
            },
            {
                "coords": (0.36, 0.14, 0.06),
                "label": "Ollama: Windy Slice",
                "color": GOLD_D,
                "winner": False,
            },
            {
                "coords": (0.12, 0.08, 0.01),
                "label": "Consensus winner",
                "color": RED_C,
                "winner": True,
            },
        ]

        label_stack = VGroup()
        for point in plot_points:
            dot = Dot3D(
                point=axes.c2p(*point["coords"]),
                radius=0.09 if point["winner"] else 0.065,
                color=point["color"],
            )
            connector = Line3D(axes.c2p(0, 0, 0), axes.c2p(*point["coords"]), color=point["color"])
            point_label = Text(point["label"], font_size=18, color=point["color"])
            point_label.next_to(origin_label, DOWN, aligned_edge=LEFT, buff=0.16)
            if len(label_stack) > 0:
                point_label.next_to(label_stack[-1], DOWN, aligned_edge=LEFT, buff=0.12)

            self.play(Create(connector), FadeIn(dot, scale=0.8), run_time=0.55)
            self.play(FadeIn(point_label, shift=RIGHT * 0.12), run_time=0.3)
            label_stack.add(point_label)

        winner_ring = Circle(radius=0.32, color=RED_C, stroke_width=5)
        winner_ring.rotate(PI / 2, axis=RIGHT)
        winner_ring.move_to(axes.c2p(0.12, 0.08, 0.01))

        self.wait(0.5)
        self.play(Create(winner_ring), run_time=1.0)
        self.play(winner_ring.animate.scale(1.18), run_time=0.45)
        self.play(winner_ring.animate.scale(1 / 1.18), run_time=0.45)
        self.begin_ambient_camera_rotation(rate=0.08)
        self.wait(3.5)
        self.stop_ambient_camera_rotation()
        self.wait(1.0)
