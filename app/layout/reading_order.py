from typing import Sequence
from app.schemas.region import Region, RegionType


class ReadingOrderSorter:
    """
    Recovers reading order across single-column and multi-column pages.
    Assigns sequential 1-based reading_order indices to all regions.
    """

    def sort_regions(
        self,
        regions: Sequence[Region],
        page_width: float,
        page_height: float,
    ) -> list[Region]:
        if not regions:
            return []

        # 1. Partition into headers, body content, and footers
        header_regions: list[Region] = []
        body_regions: list[Region] = []
        footer_regions: list[Region] = []

        for r in regions:
            if r.type in (RegionType.HEADER, RegionType.PAGE_NUMBER) and r.bbox[1] < page_height * 0.15:
                header_regions.append(r)
            elif r.type in (RegionType.FOOTER, RegionType.PAGE_NUMBER) and r.bbox[3] > page_height * 0.85:
                footer_regions.append(r)
            else:
                body_regions.append(r)

        # Sort headers top-to-bottom, left-to-right
        header_regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))

        # Sort footers top-to-bottom, left-to-right
        footer_regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))

        # 2. Sort body regions by column layout
        sorted_body = self._sort_body_content(body_regions, page_width, page_height)

        # 3. Concatenate and assign 1-based sequential reading order
        all_sorted = header_regions + sorted_body + footer_regions
        ordered_regions: list[Region] = []
        for idx, region in enumerate(all_sorted, start=1):
            r_dict = region.model_dump()
            r_dict["reading_order"] = idx
            ordered_regions.append(Region(**r_dict))

        return ordered_regions

    def order_regions(
        self,
        regions: Sequence[Region],
        page_width: float,
        page_height: float,
    ) -> list[Region]:
        """Alias for sort_regions."""
        return self.sort_regions(regions, page_width, page_height)

    def _sort_body_content(
        self,
        body_regions: list[Region],
        page_width: float,
        page_height: float,
    ) -> list[Region]:
        if not body_regions:
            return []

        is_multi_column, gutter_x = self._detect_two_columns(body_regions, page_width)

        if not is_multi_column:
            # Single-column: sort purely top-to-bottom
            return sorted(body_regions, key=lambda r: (r.bbox[1], r.bbox[0]))

        # Multi-column handling:
        left_column: list[Region] = []
        right_column: list[Region] = []
        spanning_blocks: list[Region] = []

        for r in body_regions:
            width = r.bbox[2] - r.bbox[0]
            # If region spans > 65% of page width, it is a multi-column span (e.g. title, wide figure)
            if width > page_width * 0.65:
                spanning_blocks.append(r)
            elif (r.bbox[0] + r.bbox[2]) / 2.0 < gutter_x:
                left_column.append(r)
            else:
                right_column.append(r)

        # Sort columns internally top-to-bottom
        left_sorted = sorted(left_column, key=lambda r: (r.bbox[1], r.bbox[0]))
        right_sorted = sorted(right_column, key=lambda r: (r.bbox[1], r.bbox[0]))
        spanning_sorted = sorted(spanning_blocks, key=lambda r: (r.bbox[1], r.bbox[0]))

        all_elements: list[Region] = []

        # Split columns around horizontal spanning block bands
        prev_y = 0.0
        for span in spanning_sorted:
            span_y0 = span.bbox[1]
            span_y1 = span.bbox[3]

            # Collect column elements between prev_y and span_y0
            sub_left = [r for r in left_sorted if prev_y <= r.bbox[1] < span_y0]
            sub_right = [r for r in right_sorted if prev_y <= r.bbox[1] < span_y0]
            all_elements.extend(sub_left)
            all_elements.extend(sub_right)

            # Add spanning block
            all_elements.append(span)
            prev_y = span_y1

        # Collect remaining trailing column elements
        trailing_left = [r for r in left_sorted if r.bbox[1] >= prev_y]
        trailing_right = [r for r in right_sorted if r.bbox[1] >= prev_y]
        all_elements.extend(trailing_left)
        all_elements.extend(trailing_right)

        return all_elements

    def _detect_two_columns(self, regions: list[Region], page_width: float) -> tuple[bool, float]:
        """Determine if regions exhibit a two-column distribution with a clear gutter."""
        mid_x = page_width / 2.0
        left_count = 0
        right_count = 0

        for r in regions:
            width = r.bbox[2] - r.bbox[0]
            if width > page_width * 0.65:
                continue
            center_x = (r.bbox[0] + r.bbox[2]) / 2.0
            if center_x < mid_x - (page_width * 0.05):
                left_count += 1
            elif center_x > mid_x + (page_width * 0.05):
                right_count += 1

        # If both left and right sides have non-spanning blocks
        if left_count >= 1 and right_count >= 1 and (left_count + right_count >= 2):
            return True, mid_x

        return False, mid_x


ReadingOrderDetector = ReadingOrderSorter
XYCutReadingOrderDetector = ReadingOrderSorter
