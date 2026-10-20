import { ChangeDetectionStrategy, Component, input } from '@angular/core';

@Component({
  selector: 'app-section-header',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="flex flex-wrap items-end justify-between gap-4">
      <div class="max-w-2xl">
        @if (eyebrow()) {
          <p class="label-mono mb-2">{{ eyebrow() }}</p>
        }
        <h2 class="text-2xl font-semibold tracking-[-0.035em] text-ink md:text-[28px]">
          {{ title() }}
        </h2>
        @if (description()) {
          <p class="mt-2 max-w-xl text-[13px] leading-relaxed text-muted">{{ description() }}</p>
        }
      </div>
      <ng-content select="[action]" />
    </div>
  `,
})
export class SectionHeaderComponent {
  readonly eyebrow = input<string>('');
  readonly title = input<string>('');
  readonly description = input<string>('');
}
